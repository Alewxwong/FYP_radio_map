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

# Hyperparameters
INPUT_CHANNELS = 5  
HIDDEN_CHANNELS = 32 
FNO_MODES = 16       
BATCH_SIZE = 4       
EPOCHS = 30          
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4
LAMBDA_HUBER = 1.0
LAMBDA_SOB = 0.15
OUTAGE_THRESHOLD = 0.2  # Since data is normalized to 0-1

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
        if self.file is None:
            self.file = h5py.File(self.h5_path, 'r')
        return self.file['inputs'].shape[0]

    def __getitem__(self, idx):
        if self.file is None:
            self.file = h5py.File(self.h5_path, 'r')
        
        inputs = self.file['inputs'][idx] 
        target = self.file['irt2_targets'][idx] / 255.0 # Normalize to 0-1
        return torch.tensor(inputs, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)

def get_dataloaders():
    train_ds = RadioMapH5Dataset(os.path.join(DATA_DIR, 'radiomapseer_multifidelity_train.h5'))
    val_ds = RadioMapH5Dataset(os.path.join(DATA_DIR, 'radiomapseer_multifidelity_val.h5'))
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    return train_loader, val_loader

# ==========================================
# 3. MODEL ARCHITECTURES
# ==========================================
# --- Stage 1 Components (Required to load Stage 1 weights) ---
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

# --- Stage 2 Refinement Model ---
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
        # Concatenate Stage 1 prediction with geometry priors (Building, Antenna, SDF, LoS)
        x = torch.cat([stage1_output, geometry_priors], dim=1)
        x = self.encoder(x)
        residual = self.decoder(x)
        return residual

# ==========================================
# 4. LOSS & METRIC FUNCTIONS
# ==========================================
class SobolevLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        
    def forward(self, pred, target):
        kx = self.kernel_x.to(pred.device)
        ky = self.kernel_y.to(pred.device)
        pred_grad_x = F.conv2d(pred, kx, padding=1)
        pred_grad_y = F.conv2d(pred, ky, padding=1)
        target_grad_x = F.conv2d(target, kx, padding=1)
        target_grad_y = F.conv2d(target, ky, padding=1)
        return F.mse_loss(pred_grad_x, target_grad_x) + F.mse_loss(pred_grad_y, target_grad_y)

def compute_wireless_metrics(preds, targets, outage_threshold=0.2):
    rmse = torch.sqrt(torch.mean((preds - targets) ** 2))
    pred_outage = (preds < outage_threshold).float()
    target_outage = (targets < outage_threshold).float()
    
    TP = torch.sum(pred_outage * target_outage) 
    FN = torch.sum((1 - pred_outage) * target_outage) 
    FP = torch.sum(pred_outage * (1 - target_outage)) 
    
    recall = TP / (TP + FN + 1e-8) 
    precision = TP / (TP + FP + 1e-8)
    f1_dice = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    return rmse.item(), recall.item(), f1_dice.item()

# ==========================================
# 5. STAGE 2 TRAINING LOOP
# ==========================================
def train_stage2():
    print("Loading Data...")
    train_loader, val_loader = get_dataloaders()
    
    print("Loading and Freezing Stage 1 Model...")
    stage1_model = Stage1Model(INPUT_CHANNELS, HIDDEN_CHANNELS, FNO_MODES, 1).to(device)
    if os.path.exists(STAGE1_MODEL_PATH):
        stage1_model.load_state_dict(torch.load(STAGE1_MODEL_PATH, map_location=device))
        print(f"Successfully loaded Stage 1 weights from {STAGE1_MODEL_PATH}")
    else:
        print(f"WARNING: Stage 1 weights not found at {STAGE1_MODEL_PATH}. Using random initialization.")
    
    # FREEZE Stage 1 weights! We only want to train Stage 2.
    for param in stage1_model.parameters():
        param.requires_grad = False
    stage1_model.eval()

    print("Initializing Stage 2 Refinement Model...")
    stage2_model = Stage2Refinement(in_channels=5, hidden_channels=32, out_channels=1).to(device)
    
    optimizer = torch.optim.AdamW(stage2_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    huber_loss = nn.HuberLoss(delta=1.0)
    sobolev_loss = SobolevLoss()
    
    # --- CHECKPOINT RESUME LOGIC ---
    start_epoch = 0
    best_val_f1 = -1.0
    
    if os.path.exists(CHECKPOINT_PATH):
        print(f"\n[RESUME] Checkpoint found! Loading state from {CHECKPOINT_PATH}...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        stage2_model.load_state_dict(checkpoint['stage2_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_f1 = checkpoint.get('best_val_f1', -1.0)
        print(f"[RESUME] Resuming from Epoch {start_epoch + 1} / {EPOCHS}")
        print(f"[RESUME] Previous best Outage F1: {best_val_f1:.3f}\n")
    else:
        print("\n[START] No checkpoint found. Starting Stage 2 training from scratch.\n")

    print(f"Starting Stage 2 Training for {EPOCHS} epochs...")
    
    for epoch in range(start_epoch, EPOCHS):
        # --- TRAINING PHASE ---
        stage2_model.train()
        train_loss = 0.0
        train_huber = 0.0
        train_sob = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            # 1. Get Stage 1 prediction (no gradients)
            with torch.no_grad():
                stage1_pred = stage1_model(inputs)
            
            # 2. Get Stage 2 residual
            geometry_priors = inputs[:, 0:4, :, :] 
            residual = stage2_model(stage1_pred, geometry_priors)
            
            # 3. Final prediction = Stage 1 + Residual
            final_pred = stage1_pred + residual
            
            # 4. Calculate Loss on the FINAL prediction
            loss_h = huber_loss(final_pred, targets)
            loss_s = sobolev_loss(final_pred, targets)
            loss = LAMBDA_HUBER * loss_h + LAMBDA_SOB * loss_s
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(stage2_model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_huber += loss_h.item()
            train_sob += loss_s.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
        scheduler.step()
        
        # --- VALIDATION PHASE ---
        stage2_model.eval()
        val_loss = 0.0
        val_rmse = 0.0
        val_recall = 0.0
        val_f1 = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                stage1_pred = stage1_model(inputs)
                geometry_priors = inputs[:, 0:4, :, :]
                residual = stage2_model(stage1_pred, geometry_priors)
                final_pred = stage1_pred + residual
                
                loss = huber_loss(final_pred, targets)
                val_loss += loss.item()
                
                # Calculate Wireless Metrics
                rmse, recall, f1 = compute_wireless_metrics(final_pred, targets, outage_threshold=OUTAGE_THRESHOLD)
                val_rmse += rmse
                val_recall += recall
                val_f1 += f1
                num_batches += 1
                
        avg_train_loss = train_loss / len(train_loader)
        avg_huber = train_huber / len(train_loader)
        avg_sob = train_sob / len(train_loader)
        avg_val_loss = val_loss / num_batches
        avg_val_rmse = val_rmse / num_batches
        avg_val_recall = val_recall / num_batches
        avg_val_f1 = val_f1 / num_batches
        
        print(f"Epoch {epoch+1} | Train Total: {avg_train_loss:.4f} (Huber: {avg_huber:.4f}, Sob: {avg_sob:.4f}) | "
              f"Val Loss: {avg_val_loss:.4f} | RMSE: {avg_val_rmse:.4f} | Outage Recall: {avg_val_recall:.3f} | Outage F1(Dice): {avg_val_f1:.3f}")
        
        # --- SAVE BEST MODEL (For Inference) ---
        if avg_val_f1 > best_val_f1:
            best_val_f1 = avg_val_f1
            torch.save({
                'stage1_state_dict': stage1_model.state_dict(),
                'stage2_state_dict': stage2_model.state_dict(),
            }, os.path.join(OUTPUT_DIR, 'stage2_combined_best.pth'))
            print(f"-> Saved new BEST combined model! (Outage F1: {best_val_f1:.3f})")
        
        # --- SAVE FULL CHECKPOINT (For Resuming) ---
        torch.save({
            'epoch': epoch + 1,
            'stage2_state_dict': stage2_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_f1': best_val_f1,
        }, CHECKPOINT_PATH)
        
    print("\nStage 2 Training Complete!")
    print(f"Final Best Outage F1: {best_val_f1:.3f}")

if __name__ == "__main__":
    train_stage2()