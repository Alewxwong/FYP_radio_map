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
os.makedirs(OUTPUT_DIR, exist_ok=True)

INPUT_CHANNELS = 5  
HIDDEN_CHANNELS = 32 
FNO_MODES = 16       
OUTPUT_CHANNELS = 1  
BATCH_SIZE = 4       
EPOCHS = 50          
LEARNING_RATE = 8e-4
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 3
LAMBDA_HUBER = 1.0
LAMBDA_SOB = 0.15
OUTAGE_THRESHOLD = 0.2  # Normalized scale (approx -100dBm)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================
# 2. DATASET & DATALOADER
# ==========================================
class RadioMapH5Dataset(Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path
        self.file = None 
    def __len__(self):
        if self.file is None: self.file = h5py.File(self.h5_path, 'r')
        return self.file['inputs'].shape[0]
    def __getitem__(self, idx):
        if self.file is None: self.file = h5py.File(self.h5_path, 'r')
        inputs = self.file['inputs'][idx] 
        target = self.file['irt2_targets'][idx] / 255.0 # Normalized 0-1
        return torch.tensor(inputs, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)

def get_dataloaders():
    train_ds = RadioMapH5Dataset(os.path.join(DATA_DIR, 'radiomapseer_multifidelity_train.h5'))
    val_ds = RadioMapH5Dataset(os.path.join(DATA_DIR, 'radiomapseer_multifidelity_val.h5'))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    return train_loader, val_loader

# ==========================================
# 3. STAGE 1 MODEL (FNO + Local Branch)
# ==========================================
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels; self.out_channels = out_channels
        self.modes1 = modes1; self.modes2 = modes2
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
    def forward(self, x): return self.spectral(x) + self.linear(x)

class LocalConvBranch(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.ReLU(), nn.Conv2d(out_channels, out_channels, 3, padding=1))
    def forward(self, x): return self.conv(x)

class Stage1Model(nn.Module):
    def __init__(self, input_ch, hidden_ch, modes, output_ch):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(input_ch, hidden_ch, 1), nn.ReLU())
        self.fno1 = FNOBlock(hidden_ch, hidden_ch, modes); self.local1 = LocalConvBranch(hidden_ch, hidden_ch)
        self.fno2 = FNOBlock(hidden_ch, hidden_ch, modes); self.local2 = LocalConvBranch(hidden_ch, hidden_ch)
        self.decoder = nn.Sequential(nn.Conv2d(hidden_ch, hidden_ch // 2, 1), nn.ReLU(), nn.Conv2d(hidden_ch // 2, output_ch, 1))

    def forward(self, x):
        x = self.encoder(x)
        x = x + self.fno1(x) + self.local1(x)
        x = x + self.fno2(x) + self.local2(x)
        return self.decoder(x)

# ==========================================
# 4. LOSS & METRICS
# ==========================================
class OutageAwareHuberLoss(nn.Module):
    """Punishes errors in dead zones (Outage) more heavily."""
    def __init__(self, delta=1.0, outage_threshold=0.2, outage_weight=3.0):
        super().__init__()
        self.huber = nn.HuberLoss(delta=delta, reduction='none')
        self.threshold = outage_threshold
        self.weight = outage_weight

    def forward(self, preds, targets):
        base_loss = self.huber(preds, targets)
        # Create weight map: higher weight for outage regions
        weights = torch.where(targets < self.threshold, 
                              torch.tensor(self.weight, device=preds.device), 
                              torch.tensor(1.0, device=preds.device))
        return (base_loss * weights).mean()

class SobolevLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    def forward(self, pred, target):
        kx = self.kernel_x.to(pred.device); ky = self.kernel_y.to(pred.device)
        pred_grad_x = F.conv2d(pred, kx, padding=1); pred_grad_y = F.conv2d(pred, ky, padding=1)
        target_grad_x = F.conv2d(target, kx, padding=1); target_grad_y = F.conv2d(target, ky, padding=1)
        return F.mse_loss(pred_grad_x, target_grad_x) + F.mse_loss(pred_grad_y, target_grad_y)

def compute_metrics(preds, targets, threshold=0.2):
    """Returns metrics in both Normalized (0-1) and Original (0-255) scales."""
    # Normalized Scale
    rmse_norm = torch.sqrt(torch.mean((preds - targets) ** 2)).item()
    mae_norm = torch.mean(torch.abs(preds - targets)).item()
    
    # Original Scale (Multiply by 255)
    rmse_orig = rmse_norm * 255.0
    mae_orig = mae_norm * 255.0
    
    # Outage Metrics (Threshold based)
    pred_outage = (preds < threshold).float()
    target_outage = (targets < threshold).float()
    TP = torch.sum(pred_outage * target_outage) 
    FN = torch.sum((1 - pred_outage) * target_outage) 
    FP = torch.sum(pred_outage * (1 - target_outage)) 
    recall = (TP / (TP + FN + 1e-8)).item()
    precision = (TP / (TP + FP + 1e-8)).item()
    f1 = (2 * (precision * recall) / (precision + recall + 1e-8)).item()
    
    return rmse_norm, rmse_orig, mae_orig, recall, f1

# ==========================================
# 5. TRAINING LOOP
# ==========================================
def train():
    train_loader, val_loader = get_dataloaders()
    model = Stage1Model(INPUT_CHANNELS, HIDDEN_CHANNELS, FNO_MODES, OUTPUT_CHANNELS).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS: return (epoch + 1) / WARMUP_EPOCHS
        else: return 0.5 * (1 + np.cos(np.pi * (epoch - WARMUP_EPOCHS) / (EPOCHS - WARMUP_EPOCHS)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    huber_loss = OutageAwareHuberLoss(outage_threshold=OUTAGE_THRESHOLD)
    sobolev_loss = SobolevLoss()
    
    checkpoint_path = os.path.join(OUTPUT_DIR, 'stage1_checkpoint.pth')
    best_model_path = os.path.join(OUTPUT_DIR, 'stage1_fno_best.pth')
    start_epoch, best_val_f1 = 0, -1.0
    
    if os.path.exists(checkpoint_path):
        print(f"[RESUME] Loading from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict']); optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch']; best_val_f1 = ckpt.get('best_val_f1', -1.0)
        print(f"[RESUME] Starting from Epoch {start_epoch + 1} | Best F1: {best_val_f1:.3f}\n")
    else:
        print("[START] Training from scratch.\n")

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            preds = model(inputs)
            loss = LAMBDA_HUBER * huber_loss(preds, targets) + LAMBDA_SOB * sobolev_loss(preds, targets)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss, val_rmse_norm, val_rmse_orig, val_mae_orig, val_recall, val_f1, num_batches = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                preds = model(inputs)
                val_loss += OutageAwareHuberLoss()(preds, targets).item()
                rmse_n, rmse_o, mae_o, rec, f1 = compute_metrics(preds, targets, OUTAGE_THRESHOLD)
                val_rmse_norm += rmse_n; val_rmse_orig += rmse_o; val_mae_orig += mae_o
                val_recall += rec; val_f1 += f1; num_batches += 1
                
        avg_loss = train_loss / len(train_loader)
        avg_v_loss = val_loss / num_batches
        avg_v_rmse_n = val_rmse_norm / num_batches; avg_v_rmse_o = val_rmse_orig / num_batches
        avg_v_mae_o = val_mae_orig / num_batches
        avg_v_rec = val_recall / num_batches; avg_v_f1 = val_f1 / num_batches
        
        print(f"Epoch {epoch+1} | Train: {avg_loss:.4f} | Val Loss: {avg_v_loss:.4f} | "
              f"RMSE: {avg_v_rmse_n:.4f} (Norm) / {avg_v_rmse_o:.2f} (Orig) | "
              f"MAE: {avg_v_mae_o:.2f} (Orig) | Recall: {avg_v_rec:.3f} | F1: {avg_v_f1:.3f}")
        
        if avg_v_f1 > best_val_f1:
            best_val_f1 = avg_v_f1
            torch.save(model.state_dict(), best_model_path)
            print(f"-> Saved BEST Stage 1 Model! (F1: {best_val_f1:.3f})")
            
        torch.save({'epoch': epoch + 1, 'model_state_dict': model.state_dict(), 
                    'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_f1': best_val_f1}, checkpoint_path)
    print("\nStage 1 Training Complete!")

if __name__ == "__main__":
    train()