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

# Model Hyperparameters (Must match training)
INPUT_CHANNELS = 5  
HIDDEN_CHANNELS = 32 
FNO_MODES = 16       
OUTPUT_CHANNELS = 1  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================
# 2. MODEL ARCHITECTURES
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

class Stage2Refinement(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels // 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels // 2, out_channels, 1)
        )

    def forward(self, stage1_output, geometry_priors):
        x = torch.cat([stage1_output, geometry_priors], dim=1)
        x = self.encoder(x)
        residual = self.decoder(x)
        return residual

# ==========================================
# 3. INFERENCE & VISUALIZATION
# ==========================================
def run_inference():
    print("Loading Test Data...")
    with h5py.File(TEST_FILE, 'r') as f:
        # Load 4 random samples from the test set for visualization
        num_samples = f['inputs'].shape[0]
        indices = np.random.choice(num_samples, 4, replace=False)
        indices = np.sort(indices) 
        inputs = torch.tensor(f['inputs'][indices], dtype=torch.float32)
        # Targets are normalized to 0-1 scale during training
        targets = torch.tensor(f['irt2_targets'][indices] / 255.0, dtype=torch.float32) 

    print("Loading Models...")
    # Initialize models
    stage1_model = Stage1Model(INPUT_CHANNELS, HIDDEN_CHANNELS, FNO_MODES, OUTPUT_CHANNELS).to(device)
    stage2_model = Stage2Refinement(in_channels=5, hidden_channels=HIDDEN_CHANNELS, out_channels=OUTPUT_CHANNELS).to(device)

    # Load weights
    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        stage1_model.load_state_dict(checkpoint['stage1_state_dict'])
        stage2_model.load_state_dict(checkpoint['stage2_state_dict'])
        print(f"Successfully loaded weights from {CHECKPOINT_PATH}")
    else:
        print(f"ERROR: Checkpoint not found at {CHECKPOINT_PATH}")
        return

    stage1_model.eval()
    stage2_model.eval()

    print("Running Inference...")
    with torch.no_grad():
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Stage 1 Prediction
        stage1_pred = stage1_model(inputs)

        # Stage 2 Prediction (Residual)
        geometry_priors = inputs[:, 0:4, :, :] # Building, Antenna, SDF, LoS
        residual = stage2_model(stage1_pred, geometry_priors)

        # Final Prediction
        final_pred = stage1_pred + residual

    # Move back to CPU for plotting and metric calculation
    inputs = inputs.cpu().numpy()
    targets = targets.cpu().numpy()
    stage1_pred = stage1_pred.cpu().numpy()
    final_pred = final_pred.cpu().numpy()

    # Calculate overall metrics for the 4 samples
    errors = final_pred - targets
    mse_norm = np.mean(errors ** 2)
    rmse_norm = np.sqrt(mse_norm)
    mae_norm = np.mean(np.abs(errors))
    
    # Convert back to original 0-255 scale for intuitive interpretation
    mse_orig = mse_norm * (255 ** 2)
    rmse_orig = rmse_norm * 255
    mae_orig = mae_norm * 255

    print(f"\n{'='*50}")
    print(f"Overall Metrics for 4 Visualized Samples:")
    print(f"{'='*50}")
    print(f"Normalized Scale (0.0 to 1.0):")
    print(f"  MSE : {mse_norm:.6f}")
    print(f"  RMSE: {rmse_norm:.6f}")
    print(f"  MAE : {mae_norm:.6f}")
    print(f"Original Scale (0 to 255):")
    print(f"  MSE : {mse_orig:.2f}")
    print(f"  RMSE: {rmse_orig:.2f}")
    print(f"  MAE : {mae_orig:.2f}")
    print(f"{'='*50}\n")

    # ==========================================
    # Plotting with Color Bars and Metrics
    # ==========================================
    fig, axes = plt.subplots(4, 6, figsize=(24, 16))
    plt.subplots_adjust(wspace=0.1, hspace=0.2)

    # Define colormaps and limits
    cmap_signal = 'jet'
    cmap_error = 'hot'
    vmin_signal, vmax_signal = 0.0, 1.0
    vmin_error, vmax_error = 0.0, 0.2 # Adjust based on typical error magnitude

    for i in range(4):
        # Calculate per-sample MSE and RMSE
        sample_error = final_pred[i, 0] - targets[i, 0]
        sample_mse = np.mean(sample_error ** 2)
        sample_rmse = np.sqrt(sample_mse)
        abs_error = np.abs(sample_error)

        # 1. Ground Truth
        im1 = axes[i, 0].imshow(targets[i, 0], cmap=cmap_signal, vmin=vmin_signal, vmax=vmax_signal)
        axes[i, 0].set_title(f'Ground Truth\n(Sample {i+1})', fontsize=10)
        axes[i, 0].axis('off')
        fig.colorbar(im1, ax=axes[i, 0], fraction=0.046, pad=0.04)

        # 2. Stage 1 Output
        im2 = axes[i, 1].imshow(stage1_pred[i, 0], cmap=cmap_signal, vmin=vmin_signal, vmax=vmax_signal)
        axes[i, 1].set_title('Stage 1 Output (FNO)', fontsize=10)
        axes[i, 1].axis('off')
        fig.colorbar(im2, ax=axes[i, 1], fraction=0.046, pad=0.04)

        # 3. Final Output (Stage 1 + Stage 2)
        im3 = axes[i, 2].imshow(final_pred[i, 0], cmap=cmap_signal, vmin=vmin_signal, vmax=vmax_signal)
        axes[i, 2].set_title('Final Output (Combined)', fontsize=10)
        axes[i, 2].axis('off')
        fig.colorbar(im3, ax=axes[i, 2], fraction=0.046, pad=0.04)

        # 4. Absolute Error Map (with MSE/RMSE annotated in title)
        im4 = axes[i, 3].imshow(abs_error, cmap=cmap_error, vmin=vmin_error, vmax=vmax_error)
        axes[i, 3].set_title(f'Absolute Error Map\nMSE: {sample_mse:.4f} | RMSE: {sample_rmse:.4f}', fontsize=10)
        axes[i, 3].axis('off')
        fig.colorbar(im4, ax=axes[i, 3], fraction=0.046, pad=0.04)

        # 5. Building Mask (Input Channel 0)
        axes[i, 4].imshow(inputs[i, 0], cmap='gray')
        axes[i, 4].set_title('Building Mask', fontsize=10)
        axes[i, 4].axis('off')

        # 6. SDF (Input Channel 2)
        axes[i, 5].imshow(inputs[i, 2], cmap='viridis')
        axes[i, 5].set_title('SDF (Distance to Wall)', fontsize=10)
        axes[i, 5].axis('off')

    output_image_path = os.path.join(OUTPUT_DIR, 'inference_visualization_with_metrics.png')
    plt.savefig(output_image_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {output_image_path}")
    plt.show()

if __name__ == "__main__":
    run_inference()