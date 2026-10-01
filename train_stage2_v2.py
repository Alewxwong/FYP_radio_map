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

# Hyperparameters
INPUT_CHANNELS = 5  # Building, Antenna, SDF, Tx_Dist, LoS
HIDDEN_CHANNELS = 32
FNO_MODES = 16
BATCH_SIZE = 4      # This will now easily fit in 8GB VRAM
EPOCHS = 30
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4
LAMBDA_HUBER = 1.0
LAMBDA_SOB = 0.15

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
# --- Stage 1 Classes (Required to load weights) ---
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

# --- Stage 2: Memory-Efficient Localized Refinement (Acts as Local GNO) ---
class Stage2Refinement(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=1):
        super().__init__()
        # Encoder fuses Stage 1 prediction with geometry priors (SDF, LoS, etc.)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU()
        )
        # Decoder predicts the residual (sharp edge corrections)
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels // 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels // 2, out_channels, 1)
        )

    def forward(self, stage1_output, geometry_priors):
        # Concatenate: [B, 1, H, W] + [B, 4, H, W] = [B, 5, H, W]
        x = torch.cat([stage1_output, geometry_priors], dim=1)
        x = self.encoder(x)
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
        kx = self.kernel_x.to(pred.device)
        ky = self.kernel_y.to(pred.device)
        pred_grad_x = F.conv2d(pred, kx, padding=1)
        pred_grad_y = F.conv2d(pred, ky, padding=1)
        target_grad_x = F.conv2d(target, kx, padding=1)
        target_grad_y = F.conv2d(target, ky, padding=1)
        return F.mse_loss(pred_grad_x, target_grad_x) + F.mse_loss(pred_grad_y, target_grad_y)

# ==========================================
# 5. STAGE 2 TRAINING LOOP
# ==========================================
def train_stage2():
    print("Loading Data...")
    train_loader, val_loader = get_dataloaders()
    
    print("Loading and Freezing Stage 1 Model...")
    stage1_model = Stage1Model(INPUT_CHANNELS, HIDDEN_CHANNELS, FNO_MODES, 1).to(device)
    
    # FIX: Added weights_only=True to suppress the FutureWarning
    stage1_model.load_state_dict(torch.load(STAGE1_MODEL_PATH, map_location=device, weights_only=True))
    
    # FREEZE Stage 1 weights! We only want to train Stage 2.
    for param in stage1_model.parameters():
        param.requires_grad = False
    stage1_model.eval()

    print("Initializing Stage 2 Refinement Model...")
    # Stage 2 input: 1 (Stage 1 output) + 4 (Geometry priors: Building, Antenna, SDF, LoS) = 5 channels
    stage2_model = Stage2Refinement(in_channels=5, hidden_channels=HIDDEN_CHANNELS, out_channels=1).to(device)
    
    optimizer = torch.optim.AdamW(stage2_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    huber_loss = nn.HuberLoss(delta=1.0)
    sobolev_loss = SobolevLoss()
    
    print(f"Starting Stage 2 Training for {EPOCHS} epochs...")
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        stage2_model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            # 1. Get Stage 1 prediction (no gradients)
            with torch.no_grad():
                stage1_pred = stage1_model(inputs)
            
            # 2. Get Stage 2 residual
            # inputs[:, 0:4, :, :] are the first 4 channels: Building, Antenna, SDF, LoS
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
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
        scheduler.step()
        
        # --- VALIDATION PHASE ---
        stage2_model.eval()
        val_loss = 0.0
        val_rmse = 0.0
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
                
                # Calculate RMSE
                rmse = torch.sqrt(torch.mean((final_pred - targets) ** 2))
                val_rmse += rmse.item()
                num_batches += 1
                
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / num_batches
        avg_val_rmse = val_rmse / num_batches
        
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val RMSE: {avg_val_rmse:.4f}")
        
        # Save best Stage 2 model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'stage1_state_dict': stage1_model.state_dict(),
                'stage2_state_dict': stage2_model.state_dict(),
            }, os.path.join(OUTPUT_DIR, 'stage2_combined_best.pth'))
            print("-> Saved new BEST combined model!")

    print("\nStage 2 Training Complete!")

if __name__ == "__main__":
    train_stage2()