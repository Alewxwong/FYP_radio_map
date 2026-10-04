import os
import h5py
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. CONFIGURATION
# ==========================================
DATA_DIR = r"C:\Users\user\Desktop\Fgo\dataset\processed_data"
OUTPUT_DIR = r"C:\Users\user\Desktop\Fgo\dataset\checkpoints"
TEST_FILE = os.path.join(DATA_DIR, 'radiomapseer_multifidelity_test.h5')
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, 'stage2_combined_best.pth')

INPUT_CHANNELS = 5  
HIDDEN_CHANNELS = 32 
FNO_MODES = 16       

# Metric Conversion (Matches your training logs)
DB_RANGE = 139.0 
PIXEL_MAX = 255.0
DB_FACTOR = DB_RANGE / PIXEL_MAX  # ~0.545 dB per pixel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================
# 2. MODEL ARCHITECTURES (Must match training exactly)
# ==========================================
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))

class FNOBlock(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.spectral = SpectralConv2d(in_channels, out_channels, modes, modes)
        self.linear = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.spectral(x) + self.linear(x)

class LocalConvBranch(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )

    def forward(self, x):
        return self.conv(x)

class Stage1Model(nn.Module):
    def __init__(self, input_ch, hidden_ch, modes, output_ch):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(input_ch, hidden_ch, 1), nn.ReLU())
        self.fno1 = FNOBlock(hidden_ch, hidden_ch, modes)
        self.fno2 = FNOBlock(hidden_ch, hidden_ch, modes)
        self.local1 = LocalConvBranch(hidden_ch, hidden_ch)
        self.local2 = LocalConvBranch(hidden_ch, hidden_ch)
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_ch, hidden_ch // 2, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_ch // 2, output_ch, 1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x + self.fno1(x) + self.local1(x)
        x = x + self.fno2(x) + self.local2(x)
        x = self.decoder(x)
        return x

# --- Upgraded Stage 2: Diffraction-Aware Refinement ---
class DiffractionRefinement(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU()
        )
        self.dilated_block = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=4, dilation=4),
            nn.ReLU()
        )
        self.gate_conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Sigmoid()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels // 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels // 2, out_channels, 1)
        )

    def forward(self, stage1_output, geometry_priors):
        x = torch.cat([stage1_output, geometry_priors], dim=1)
        x = self.encoder(x)
        x = self.dilated_block(x)
        
        # SDF is channel index 2 in geometry_priors (Building=0, Antenna=1, SDF=2, LoS=3)
        sdf = geometry_priors[:, 2:3, :, :]
        edge_gate = self.gate_conv(sdf)
        
        x = x * edge_gate
        residual = self.decoder(x)
        return residual

# ==========================================
# 3. INFERENCE & VISUALIZATION
# ==========================================
def run_inference():
    print("Loading Test Data...")
    with h5py.File(TEST_FILE, 'r') as f:
        num_samples = f['inputs'].shape[0]
        indices = np.random.choice(num_samples, 4, replace=False)
        
        # FIX 1: Sort indices to prevent h5py "Indexing elements must be in increasing order" error
        indices = np.sort(indices) 
        
        inputs = torch.tensor(f['inputs'][indices], dtype=torch.float32)
        targets = torch.tensor(f['irt2_targets'][indices] / 255.0, dtype=torch.float32) # Normalized 0-1

    print("Loading Models...")
    stage1_model = Stage1Model(INPUT_CHANNELS, HIDDEN_CHANNELS, FNO_MODES, 1).to(device)
    stage2_model = DiffractionRefinement(in_channels=5, hidden_channels=HIDDEN_CHANNELS, out_channels=1).to(device)

    if os.path.exists(CHECKPOINT_PATH):
        # FIX 2: weights_only=False suppresses the PyTorch security warning
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
        stage1_model.load_state_dict(checkpoint['stage1_state_dict'])
        stage2_model.load_state_dict(checkpoint['stage2_state_dict'])
        print("Successfully loaded Stage 1 and Stage 2 weights.")
    else:
        print(f"ERROR: Checkpoint not found at {CHECKPOINT_PATH}")
        return

    stage1_model.eval()
    stage2_model.eval()

    print("Running Inference...")
    with torch.no_grad():
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        stage1_pred = stage1_model(inputs)
        geometry_priors = inputs[:, 0:4, :, :]
        residual = stage2_model(stage1_pred, geometry_priors)
        final_pred = stage1_pred + residual

    # Move back to CPU for metric calculation and plotting
    inputs = inputs.cpu().numpy()
    targets = targets.cpu().numpy()
    stage1_pred = stage1_pred.cpu().numpy()
    final_pred = final_pred.cpu().numpy()

    # ==========================================
    # Calculate Metrics (in dB to match training logs)
    # ==========================================
    rmse_norm = np.sqrt(np.mean((final_pred - targets) ** 2))
    mae_norm = np.mean(np.abs(final_pred - targets))
    
    # Convert to dB
    rmse_db = rmse_norm * PIXEL_MAX * DB_FACTOR
    mae_db = mae_norm * PIXEL_MAX * DB_FACTOR
    
    # Outage Metrics (Threshold = 0.2 normalized)
    outage_threshold = 0.2
    pred_outage = (final_pred < outage_threshold).astype(float)
    target_outage = (targets < outage_threshold).astype(float)
    
    TP = np.sum(pred_outage * target_outage)
    FN = np.sum((1 - pred_outage) * target_outage)
    FP = np.sum(pred_outage * (1 - target_outage))
    
    recall = TP / (TP + FN + 1e-8)
    precision = TP / (TP + FP + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

    print("\n" + "="*60)
    print("OVERALL TEST SET METRICS (4 Random Samples)")
    print("="*60)
    print(f"Normalized Scale (0.0 to 1.0):")
    print(f"  RMSE: {rmse_norm:.4f}")
    print(f"  MAE : {mae_norm:.4f}")
    print(f"Decibel Scale (dB) - Matches Training Logs:")
    print(f"  RMSE: {rmse_db:.2f} dB")
    print(f"  MAE : {mae_db:.2f} dB")
    print(f"Outage Detection (Threshold < 0.2):")
    print(f"  Recall    : {recall:.3f}")
    print(f"  Precision : {precision:.3f}")
    print(f"  Outage F1 : {f1:.3f}")
    print("="*60 + "\n")

    # ==========================================
    # Visualization
    # ==========================================
    fig, axes = plt.subplots(4, 7, figsize=(28, 16))
    plt.subplots_adjust(wspace=0.1, hspace=0.2)

    for i in range(4):
        # 1. Building Mask
        axes[i, 0].imshow(inputs[i, 0], cmap='gray')
        axes[i, 0].set_title(f'Sample {i+1}\nBuilding Mask', fontsize=10)
        axes[i, 0].axis('off')

        # 2. SDF
        axes[i, 1].imshow(inputs[i, 2], cmap='viridis')
        axes[i, 1].set_title('SDF', fontsize=10)
        axes[i, 1].axis('off')

        # 3. Ground Truth
        im_gt = axes[i, 2].imshow(targets[i, 0], cmap='jet', vmin=0, vmax=1)
        axes[i, 2].set_title('Ground Truth', fontsize=10)
        axes[i, 2].axis('off')
        fig.colorbar(im_gt, ax=axes[i, 2], fraction=0.046, pad=0.04)

        # 4. Stage 1 Output
        im_s1 = axes[i, 3].imshow(stage1_pred[i, 0], cmap='jet', vmin=0, vmax=1)
        axes[i, 3].set_title('Stage 1 (FNO)', fontsize=10)
        axes[i, 3].axis('off')
        fig.colorbar(im_s1, ax=axes[i, 3], fraction=0.046, pad=0.04)

        # 5. Final Output (Stage 1 + Stage 2)
        im_final = axes[i, 4].imshow(final_pred[i, 0], cmap='jet', vmin=0, vmax=1)
        axes[i, 4].set_title('Final (FNO + GNO)', fontsize=10)
        axes[i, 4].axis('off')
        fig.colorbar(im_final, ax=axes[i, 4], fraction=0.046, pad=0.04)

        # 6. Absolute Error Map
        error_map = np.abs(targets[i, 0] - final_pred[i, 0])
        im_err = axes[i, 5].imshow(error_map, cmap='hot', vmin=0, vmax=0.2)
        axes[i, 5].set_title('Absolute Error', fontsize=10)
        axes[i, 5].axis('off')
        fig.colorbar(im_err, ax=axes[i, 5], fraction=0.046, pad=0.04)
        
        # 7. Error Map Zoom (Bottom Right Corner example)
        axes[i, 6].imshow(error_map[180:256, 180:256], cmap='hot', vmin=0, vmax=0.2)
        axes[i, 6].set_title('Error Zoom\n(Corner)', fontsize=10)
        axes[i, 6].axis('off')

    output_image_path = os.path.join(OUTPUT_DIR, 'inference_visualization.png')
    plt.savefig(output_image_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {output_image_path}")
    plt.show()

if __name__ == "__main__":
    run_inference()