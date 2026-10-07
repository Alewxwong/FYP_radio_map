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

INPUT_CHANNELS = 5; HIDDEN_CHANNELS = 32; FNO_MODES = 16; BATCH_SIZE = 4
EPOCHS = 30; LEARNING_RATE = 5e-4; WEIGHT_DECAY = 1e-4
LAMBDA_HUBER = 1.0; LAMBDA_SOB = 0.15; OUTAGE_THRESHOLD = 0.2

# --- METRIC CONVERSION ---
DB_RANGE = 139.0 
PIXEL_MAX = 255.0
DB_FACTOR = DB_RANGE / PIXEL_MAX  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. DATASET & DATALOADER
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
    mae_norm = torch.mean(torch.abs(preds - targets)).item()
    rmse_db = rmse_norm * PIXEL_MAX * DB_FACTOR
    mae_db = mae_norm * PIXEL_MAX * DB_FACTOR
    
    pred_outage = (preds < threshold).float(); target_outage = (targets < threshold).float()
    TP = torch.sum(pred_outage * target_outage); FN = torch.sum((1 - pred_outage) * target_outage); FP = torch.sum(pred_outage * (1 - target_outage)) 
    recall = (TP / (TP + FN + 1e-8)).item(); precision = (TP / (TP + FP + 1e-8)).item()
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    return rmse_db, mae_db, recall, f1

# ==========================================
# 3. MODELS (Stage 1 + Diffraction-Aware Stage 2)
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

# --- Stage 2: Diffraction-Aware Refinement Block ---
class DiffractionRefinement(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=1):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(in_channels, hidden_channels, 3, padding=1), nn.ReLU(), nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1), nn.ReLU())
        self.dilated_block = nn.Sequential(nn.Conv2d(hidden_channels, hidden_channels, 3, padding=2, dilation=2), nn.ReLU(), nn.Conv2d(hidden_channels, hidden_channels, 3, padding=4, dilation=4), nn.ReLU())
        self.gate_conv = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 1, 3, padding=1), nn.Sigmoid())
        self.decoder = nn.Sequential(nn.Conv2d(hidden_channels, hidden_channels // 2, 3, padding=1), nn.ReLU(), nn.Conv2d(hidden_channels // 2, out_channels, 1))

    def forward(self, stage1_output, geometry_priors):
        sdf = geometry_priors[:, 2:3, :, :]
        x = torch.cat([stage1_output, geometry_priors], dim=1)
        x = self.encoder(x)
        x = self.dilated_block(x)
        edge_gate = self.gate_conv(sdf)
        x = x * edge_gate
        residual = self.decoder(x)
        return residual

# ==========================================
# 4. LOSS FUNCTIONS
# ==========================================
class ShadowAwareLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.huber = nn.HuberLoss(delta=1.0, reduction='none')
    def forward(self, pred, target, los_mask, sdf):
        base_loss = self.huber(pred, target)
        shadow_weight = 1.0 + 3.0 * (1.0 - los_mask) 
        edge_weight = 1.0 + 4.0 * torch.exp(-sdf * 15.0)
        final_weight = shadow_weight * edge_weight
        return (base_loss * final_weight).mean()

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
    
    stage1_model = Stage1Model(INPUT_CHANNELS, HIDDEN_CHANNELS, FNO_MODES, 1).to(device)
    stage1_model.load_state_dict(torch.load(STAGE1_MODEL_PATH, map_location=device, weights_only=True))
    for param in stage1_model.parameters(): param.requires_grad = False
    stage1_model.eval()

    stage2_model = DiffractionRefinement(in_channels=5, hidden_channels=32, out_channels=1).to(device)
    optimizer = torch.optim.AdamW(stage2_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    shadow_aware_loss = ShadowAwareLoss()
    sobolev_loss = SobolevLoss()
    
    checkpoint_path = os.path.join(OUTPUT_DIR, 'stage2_checkpoint.pth')
    best_model_path = os.path.join(OUTPUT_DIR, 'stage2_combined_best.pth')
    start_epoch, best_val_f1 = 0, -1.0
    
    if os.path.exists(checkpoint_path):
        print(f"[RESUME] Loading Stage 2 from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        stage2_model.load_state_dict(ckpt['stage2_state_dict']); optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch']; best_val_f1 = ckpt.get('best_val_f1', -1.0)
        print(f"[RESUME] Starting from Epoch {start_epoch + 1} | Best F1: {best_val_f1:.3f}\n")
    else:
        print("[START] Training Stage 2 from scratch.\n")

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
            
            los_mask = geometry_priors[:, 3:4, :, :]
            sdf = geometry_priors[:, 2:3, :, :]
            
            loss_shadow = shadow_aware_loss(final_pred, targets, los_mask, sdf)
            loss_sob = sobolev_loss(final_pred, targets)
            loss = LAMBDA_HUBER * loss_shadow + LAMBDA_SOB * loss_sob
            
            loss.backward(); torch.nn.utils.clip_grad_norm_(stage2_model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        scheduler.step()
        
        # Validation
        stage2_model.eval()
        val_loss, val_rmse_db, val_mae_db, val_rec, val_f1, num_batches = 0.0, 0.0, 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                stage1_pred = stage1_model(inputs)
                geometry_priors = inputs[:, 0:4, :, :]
                residual = stage2_model(stage1_pred, geometry_priors)
                final_pred = stage1_pred + residual
                
                val_loss += F.huber_loss(final_pred, targets, delta=1.0).item()
                rmse_db, mae_db, rec, f1 = compute_metrics(final_pred, targets, OUTAGE_THRESHOLD)
                val_rmse_db += rmse_db; val_mae_db += mae_db
                val_rec += rec; val_f1 += f1; num_batches += 1
                
        avg_loss = train_loss / len(train_loader)
        avg_v_loss = val_loss / num_batches
        avg_v_rmse_db = val_rmse_db / num_batches
        avg_v_mae_db = val_mae_db / num_batches
        avg_v_rec = val_rec / num_batches; avg_v_f1 = val_f1 / num_batches
        
        print(f"Epoch {epoch+1} | Train Loss: {avg_loss:.4f} | Val Loss: {avg_v_loss:.4f} | "
              f"RMSE: {avg_v_rmse_db:.2f} dB | MAE: {avg_v_mae_db:.2f} dB | "
              f"Recall: {avg_v_rec:.3f} | F1: {avg_v_f1:.3f}")
        
        if avg_v_f1 > best_val_f1:
            best_val_f1 = avg_v_f1
            torch.save({'stage1_state_dict': stage1_model.state_dict(), 'stage2_state_dict': stage2_model.state_dict()}, best_model_path)
            print(f"-> Saved BEST Combined Model! (F1: {best_val_f1:.3f})")
            
        torch.save({'epoch': epoch + 1, 'stage2_state_dict': stage2_model.state_dict(), 
                    'optimizer_state_dict': optimizer.state_dict(), 'best_val_f1': best_val_f1}, checkpoint_path)
    print("\nStage 2 Training Complete!")

if __name__ == "__main__":
    train_stage2()