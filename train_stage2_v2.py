import os
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ==========================================
# 1. CONFIGURATION
# ==========================================
DATA_DIR = r"C:\Users\user\Desktop\Fgo\dataset\processed_data"
OUTPUT_DIR = r"C:\Users\user\Desktop\Fgo\dataset\checkpoints"
STAGE1_MODEL_PATH = os.path.join(OUTPUT_DIR, 'stage1_fno_best.pth')
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, 'stage2_checkpoint.pth')

INPUT_CHANNELS = 5; HIDDEN_CHANNELS = 32; FNO_MODES = 16; BATCH_SIZE = 4
EPOCHS = 30; LEARNING_RATE = 5e-4; WEIGHT_DECAY = 1e-4
LAMBDA_HUBER = 1.0; LAMBDA_SOB = 0.15; OUTAGE_THRESHOLD = 0.2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. DATASET & METRICS (Same as Stage 1)
# ==========================================
class RadioMapH5Dataset(Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path; self.file = None 
    def __len__(self):
        if self.file is None: self.file = h5py.File(self.h5_path, 'r')
        return self.file['inputs'].shape[0]
    def __getitem__(self, idx):
        if self.file is None: self.file = h5py.File(self.h5_path, 'r')
        return torch.tensor(self.file['inputs'][idx], dtype=torch.float32), torch.tensor(self.file['irt2_targets'][idx] / 255.0, dtype=torch.float32)

def get_dataloaders():
    train_ds = RadioMapH5Dataset(os.path.join(DATA_DIR, 'radiomapseer_multifidelity_train.h5'))
    val_ds = RadioMapH5Dataset(os.path.join(DATA_DIR, 'radiomapseer_multifidelity_val.h5'))
    return DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0), DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

def compute_metrics(preds, targets, threshold=0.2):
    rmse_norm = torch.sqrt(torch.mean((preds - targets) ** 2)).item()
    rmse_orig = rmse_norm * 255.0; mae_orig = torch.mean(torch.abs(preds - targets)).item() * 255.0
    pred_outage = (preds < threshold).float(); target_outage = (targets < threshold).float()
    TP = torch.sum(pred_outage * target_outage); FN = torch.sum((1 - pred_outage) * target_outage); FP = torch.sum(pred_outage * (1 - target_outage)) 
    recall = (TP / (TP + FN + 1e-8)).item(); precision = (TP / (TP + FP + 1e-8)).item()
    f1 = (2 * (precision * recall) / (precision + recall + 1e-8)).item()
    return rmse_norm, rmse_orig, mae_orig, recall, f1

# ==========================================
# 3. MODELS (Stage 1 + GNO Stage 2)
# ==========================================
# --- Stage 1 Classes (Required to load weights) ---
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__(); self.in_channels = in_channels; self.out_channels = out_channels
        self.modes1 = modes1; self.modes2 = modes2
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
    def compl_mul2d(self, input, weights): return torch.einsum("bixy,ioxy->boxy", input, weights)
    def forward(self, x):
        batchsize = x.shape[0]; x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))

class FNOBlock(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super().__init__(); self.spectral = SpectralConv2d(in_channels, out_channels, modes, modes); self.linear = nn.Conv2d(in_channels, out_channels, 1)
    def forward(self, x): return self.spectral(x) + self.linear(x)

class LocalConvBranch(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__(); self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.ReLU(), nn.Conv2d(out_channels, out_channels, 3, padding=1))
    def forward(self, x): return self.conv(x)

class Stage1Model(nn.Module):
    def __init__(self, input_ch, hidden_ch, modes, output_ch):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(input_ch, hidden_ch, 1), nn.ReLU())
        self.fno1 = FNOBlock(hidden_ch, hidden_ch, modes); self.local1 = LocalConvBranch(hidden_ch, hidden_ch)
        self.fno2 = FNOBlock(hidden_ch, hidden_ch, modes); self.local2 = LocalConvBranch(hidden_ch, hidden_ch)
        self.decoder = nn.Sequential(nn.Conv2d(hidden_ch, hidden_ch // 2, 1), nn.ReLU(), nn.Conv2d(hidden_ch // 2, output_ch, 1))
    def forward(self, x):
        x = self.encoder(x); x = x + self.fno1(x) + self.local1(x); x = x + self.fno2(x) + self.local2(x)
        return self.decoder(x)

# --- Stage 2 GNO (Edge Diffraction Module) ---
class EdgeDiffractionGNO(nn.Module):
    """
    Simulates Graph Neural Operator (GNO) for Diffraction.
    Uses SDF to gate a Local Self-Attention mechanism, allowing information 
    to propagate along building edges and shadow boundaries.
    """
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.query_conv = nn.Conv2d(in_channels, hidden_channels // 8, 1)
        self.key_conv = nn.Conv2d(in_channels, hidden_channels // 8, 1)
        self.value_conv = nn.Conv2d(in_channels, hidden_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        # SDF Gate: Highlights edges where SDF is low (close to wall)
        self.sdf_gate = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 1, 1), nn.Sigmoid())

    def forward(self, x, sdf_channel):
        # 1. Calculate Edge Attention Mask from SDF
        edge_mask = self.sdf_gate(sdf_channel) 
        
        # 2. Local Self-Attention (Simulating Graph Message Passing)
        B, C, H, W = x.size()
        proj_query = self.query_conv(x).view(B, -1, H * W).permute(0, 2, 1) 
        proj_key = self.key_conv(x).view(B, -1, H * W)
        energy = torch.bmm(proj_query, proj_key) 
        attention = F.softmax(energy, dim=-1) 
        
        proj_value = self.value_conv(x).view(B, -1, H * W)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)
        
        # 3. Gated Residual: Only apply GNO correction at edges/shadows
        out = self.gamma * out * edge_mask + x
        return out

class Stage2Model(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        # Encoder takes Stage 1 output + 4 Geometry Priors = 5 channels
        self.encoder = nn.Sequential(nn.Conv2d(in_channels, hidden_channels, 3, padding=1), nn.ReLU())
        self.gno_block = EdgeDiffractionGNO(hidden_channels, hidden_channels)
        self.decoder = nn.Sequential(nn.Conv2d(hidden_channels, hidden_channels // 2, 3, padding=1), nn.ReLU(), nn.Conv2d(hidden_channels // 2, 1, 1))

    def forward(self, stage1_pred, geometry_priors):
        # geometry_priors: [B, 4, H, W] (Building, Antenna, SDF, LoS)
        # Extract SDF (Channel 2) for the GNO gate
        sdf_channel = geometry_priors[:, 2:3, :, :] 
        x = torch.cat([stage1_pred, geometry_priors], dim=1)
        x = self.encoder(x)
        x = self.gno_block(x, sdf_channel)
        residual = self.decoder(x)
        return residual

# ==========================================
# 4. LOSS FUNCTIONS
# ==========================================
class SobolevLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    def forward(self, pred, target):
        kx = self.kernel_x.to(pred.device); ky = self.kernel_y.to(pred.device)
        return F.mse_loss(F.conv2d(pred, kx, padding=1), F.conv2d(target, kx, padding=1)) + \
               F.mse_loss(F.conv2d(pred, ky, padding=1), F.conv2d(target, ky, padding=1))

# ==========================================
# 5. TRAINING LOOP
# ==========================================
def train_stage2():
    train_loader, val_loader = get_dataloaders()
    
    # Load and Freeze Stage 1
    stage1_model = Stage1Model(INPUT_CHANNELS, HIDDEN_CHANNELS, FNO_MODES, 1).to(device)
    stage1_model.load_state_dict(torch.load(STAGE1_MODEL_PATH, map_location=device))
    for param in stage1_model.parameters(): param.requires_grad = False
    stage1_model.eval()

    # Init Stage 2 GNO
    stage2_model = Stage2Model(in_channels=5, hidden_channels=32).to(device)
    optimizer = torch.optim.AdamW(stage2_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    huber_loss = nn.HuberLoss(delta=1.0); sobolev_loss = SobolevLoss()
    
    start_epoch, best_val_f1 = 0, -1.0
    if os.path.exists(CHECKPOINT_PATH):
        print(f"[RESUME] Loading Stage 2 from {CHECKPOINT_PATH}...")
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
        stage2_model.load_state_dict(ckpt['stage2_state_dict']); optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch']; best_val_f1 = ckpt.get('best_val_f1', -1.0)
        print(f"[RESUME] Starting from Epoch {start_epoch + 1} | Best F1: {best_val_f1:.3f}\n")
    else:
        print("[START] Training Stage 2 GNO from scratch.\n")

    for epoch in range(start_epoch, EPOCHS):
        stage2_model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            with torch.no_grad(): stage1_pred = stage1_model(inputs)
            geometry_priors = inputs[:, 0:4, :, :] 
            residual = stage2_model(stage1_pred, geometry_priors)
            final_pred = stage1_pred + residual
            
            loss = LAMBDA_HUBER * huber_loss(final_pred, targets) + LAMBDA_SOB * sobolev_loss(final_pred, targets)
            loss.backward(); torch.nn.utils.clip_grad_norm_(stage2_model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        scheduler.step()
        
        # Validation
        stage2_model.eval()
        val_loss, val_rmse_n, val_rmse_o, val_mae_o, val_rec, val_f1, num_batches = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                stage1_pred = stage1_model(inputs)
                geometry_priors = inputs[:, 0:4, :, :]
                residual = stage2_model(stage1_pred, geometry_priors)
                final_pred = stage1_pred + residual
                
                val_loss += huber_loss(final_pred, targets).item()
                rmse_n, rmse_o, mae_o, rec, f1 = compute_metrics(final_pred, targets, OUTAGE_THRESHOLD)
                val_rmse_n += rmse_n; val_rmse_o += rmse_o; val_mae_o += mae_o
                val_rec += rec; val_f1 += f1; num_batches += 1
                
        avg_loss = train_loss / len(train_loader)
        avg_v_loss = val_loss / num_batches
        avg_v_rmse_n = val_rmse_n / num_batches; avg_v_rmse_o = val_rmse_o / num_batches
        avg_v_mae_o = val_mae_o / num_batches; avg_v_rec = val_rec / num_batches; avg_v_f1 = val_f1 / num_batches
        
        print(f"Epoch {epoch+1} | Train: {avg_loss:.4f} | Val Loss: {avg_v_loss:.4f} | "
              f"RMSE: {avg_v_rmse_n:.4f} (Norm) / {avg_v_rmse_o:.2f} (Orig) | "
              f"MAE: {avg_v_mae_o:.2f} (Orig) | Recall: {avg_v_rec:.3f} | F1: {avg_v_f1:.3f}")
        
        if avg_v_f1 > best_val_f1:
            best_val_f1 = avg_v_f1
            torch.save({'stage1_state_dict': stage1_model.state_dict(), 'stage2_state_dict': stage2_model.state_dict()}, 
                       os.path.join(OUTPUT_DIR, 'stage2_combined_best.pth'))
            print(f"-> Saved BEST Combined Model! (F1: {best_val_f1:.3f})")
            
        torch.save({'epoch': epoch + 1, 'stage2_state_dict': stage2_model.state_dict(), 
                    'optimizer_state_dict': optimizer.state_dict(), 'best_val_f1': best_val_f1}, CHECKPOINT_PATH)
    print("\nStage 2 GNO Training Complete!")

if __name__ == "__main__":
    train_stage2()