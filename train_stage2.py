import os
import math
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import (
    get_device,
    RadioMapH5Dataset,
    Stage1Model,
    Stage2Model,
    RadioMapLoss,
    MetricAccumulator,
    BUILDING_CH,
    LOS_CH,
)


# ==========================================================
# Configuration
# ==========================================================
DATA_DIR = r"C:\Users\user\Desktop\Fgo\dataset\processed_data"
OUTPUT_DIR = r"C:\Users\user\Desktop\Fgo\dataset\checkpoints"

TRAIN_H5 = os.path.join(DATA_DIR, "radiomapseer_multifidelity_train.h5")
VAL_H5 = os.path.join(DATA_DIR, "radiomapseer_multifidelity_val.h5")

STAGE1_BEST_PATH = os.path.join(OUTPUT_DIR, "stage1_fno_best.pth")

BEST_COMBINED_PATH = os.path.join(OUTPUT_DIR, "stage2_combined_best.pth")
LAST_STAGE2_PATH = os.path.join(OUTPUT_DIR, "stage2_last.pth")

INPUT_CHANNELS = 5

# These must match your trained Stage 1 model.
STAGE1_BASE_CHANNELS = 32
STAGE1_FNO_MODES = 16

# Stage 2 size.
STAGE2_BASE_CHANNELS = 32
STAGE2_FNO_MODES = 16

BATCH_SIZE = 4
ACCUM_STEPS = 4
NUM_WORKERS = 0
USE_AMP = False  # Keep False for stable FFT training.

# Phase 1: train Stage 2 only, Stage 1 frozen.
STAGE2_EPOCHS = 40
STAGE2_LR = 3e-4
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 3

# Phase 2: optional joint fine-tuning.
JOINT_FINETUNE = True
JOINT_EPOCHS = 10
JOINT_STAGE1_LR = 1e-5
JOINT_STAGE2_LR = 5e-5
JOINT_WARMUP_EPOCHS = 1

# Loss weights.
L1_WEIGHT = 1.0
MSE_WEIGHT = 0.5
SOBOLEV_WEIGHT = 0.10
FFT_WEIGHT = 0.02
OUTAGE_WEIGHT = 0.5
SHADOW_WEIGHT = 0.5
EDGE_WEIGHT = 0.5

CONFIG = {
    "input_ch": INPUT_CHANNELS,
    "stage1_base": STAGE1_BASE_CHANNELS,
    "stage1_modes": STAGE1_FNO_MODES,
    "stage2_base": STAGE2_BASE_CHANNELS,
    "stage2_modes": STAGE2_FNO_MODES,
}


def save_combined(path, stage1_model, stage2_model, best_rmse, epoch):
    torch.save(
        {
            "stage1_state_dict": stage1_model.state_dict(),
            "stage2_state_dict": stage2_model.state_dict(),
            "best_rmse": best_rmse,
            "epoch": epoch,
            "config": CONFIG,
        },
        path,
    )


def evaluate_combined(stage1_model, stage2_model, val_loader, criterion, device, use_amp):
    stage1_model.eval()
    stage2_model.eval()

    metrics = MetricAccumulator()

    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, enabled=use_amp):
                stage1_pred = stage1_model(inputs)
                final_pred = stage2_model(stage1_pred, inputs)

                los_mask = inputs[:, LOS_CH:LOS_CH + 1]
                edge_mask = stage2_model.edge_extractor(inputs[:, BUILDING_CH:BUILDING_CH + 1])

                loss = criterion(
                    final_pred,
                    targets,
                    los_mask=los_mask,
                    edge_mask=edge_mask,
                )

            total_loss += loss.item()
            num_batches += 1
            metrics.update(final_pred, targets)

    out = metrics.compute()
    out["loss"] = total_loss / max(1, num_batches)
    return out


def train_stage2_phase1(stage1_model, train_loader, val_loader, device):
    print("\n" + "=" * 70)
    print("Stage 2 Phase 1: Training Stage 2 only, Stage 1 frozen")
    print("=" * 70)

    # Freeze Stage 1.
    for p in stage1_model.parameters():
        p.requires_grad = False
    stage1_model.eval()

    stage2_model = Stage2Model(
        input_ch=INPUT_CHANNELS,
        base=STAGE2_BASE_CHANNELS,
        fno_modes=STAGE2_FNO_MODES,
    ).to(device)

    criterion = RadioMapLoss(
        l1_weight=L1_WEIGHT,
        mse_weight=MSE_WEIGHT,
        sobolev_weight=SOBOLEV_WEIGHT,
        fft_weight=FFT_WEIGHT,
        outage_weight=OUTAGE_WEIGHT,
        shadow_weight=SHADOW_WEIGHT,
        edge_weight=EDGE_WEIGHT,
    ).to(device)

    optimizer = torch.optim.AdamW(
        stage2_model.parameters(),
        lr=STAGE2_LR,
        weight_decay=WEIGHT_DECAY,
    )

    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            return float(epoch + 1) / float(max(1, WARMUP_EPOCHS))

        progress = (epoch - WARMUP_EPOCHS) / max(1, STAGE2_EPOCHS - WARMUP_EPOCHS)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    use_amp = USE_AMP and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_rmse = float("inf")

    for epoch in range(STAGE2_EPOCHS):
        stage2_model.train()
        stage1_model.eval()

        optimizer.zero_grad()

        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Stage2 Epoch {epoch + 1}/{STAGE2_EPOCHS} [Train]")

        for step, (inputs, targets) in enumerate(pbar):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.no_grad():
                stage1_pred = stage1_model(inputs)

            with torch.autocast(device_type=device.type, enabled=use_amp):
                final_pred = stage2_model(stage1_pred, inputs)

                los_mask = inputs[:, LOS_CH:LOS_CH + 1]
                edge_mask = stage2_model.edge_extractor(inputs[:, BUILDING_CH:BUILDING_CH + 1])

                loss = criterion(
                    final_pred,
                    targets,
                    los_mask=los_mask,
                    edge_mask=edge_mask,
                )

                loss = loss / ACCUM_STEPS

            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if ((step + 1) % ACCUM_STEPS == 0) or ((step + 1) == len(train_loader)):
                if use_amp:
                    scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(stage2_model.parameters(), max_norm=1.0)

                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad()

            train_loss += loss.item() * ACCUM_STEPS
            pbar.set_postfix({"loss": f"{loss.item() * ACCUM_STEPS:.5f}"})

        scheduler.step()
        train_loss /= len(train_loader)

        val_out = evaluate_combined(
            stage1_model,
            stage2_model,
            val_loader,
            criterion,
            device,
            use_amp,
        )

        print(
            f"Stage2 Epoch {epoch + 1} | "
            f"Train Loss: {train_loss:.5f} | "
            f"Val Loss: {val_out['loss']:.5f} | "
            f"RMSE: {val_out['rmse_db']:.3f} dB | "
            f"MAE: {val_out['mae_db']:.3f} dB | "
            f"Recall: {val_out['recall']:.3f} | "
            f"F1: {val_out['f1']:.3f}"
        )

        if val_out["rmse_db"] < best_rmse:
            best_rmse = val_out["rmse_db"]
            save_combined(BEST_COMBINED_PATH, stage1_model, stage2_model, best_rmse, epoch + 1)
            print(f"-> Saved BEST combined model! RMSE: {best_rmse:.3f} dB")

        torch.save(
            {
                "epoch": epoch + 1,
                "stage2_state_dict": stage2_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_rmse": best_rmse,
                "config": CONFIG,
            },
            LAST_STAGE2_PATH,
        )

    # Load best Stage 2 weights before joint fine-tuning.
    if os.path.exists(BEST_COMBINED_PATH):
        print(f"\nLoading best combined checkpoint from {BEST_COMBINED_PATH}")
        ckpt = torch.load(BEST_COMBINED_PATH, map_location=device, weights_only=False)
        stage1_model.load_state_dict(ckpt["stage1_state_dict"])
        stage2_model.load_state_dict(ckpt["stage2_state_dict"])
        best_rmse = ckpt.get("best_rmse", best_rmse)

    return stage2_model, best_rmse


def joint_finetune(stage1_model, stage2_model, train_loader, val_loader, device, best_rmse):
    print("\n" + "=" * 70)
    print("Stage 2 Phase 2: Joint fine-tuning Stage 1 + Stage 2")
    print("=" * 70)

    # Unfreeze Stage 1.
    for p in stage1_model.parameters():
        p.requires_grad = True

    criterion = RadioMapLoss(
        l1_weight=L1_WEIGHT,
        mse_weight=MSE_WEIGHT,
        sobolev_weight=SOBOLEV_WEIGHT,
        fft_weight=FFT_WEIGHT,
        outage_weight=OUTAGE_WEIGHT,
        shadow_weight=SHADOW_WEIGHT,
        edge_weight=EDGE_WEIGHT,
    ).to(device)

    optimizer = torch.optim.AdamW(
        [
            {"params": stage1_model.parameters(), "lr": JOINT_STAGE1_LR},
            {"params": stage2_model.parameters(), "lr": JOINT_STAGE2_LR},
        ],
        weight_decay=WEIGHT_DECAY,
    )

    def lr_lambda(epoch):
        if epoch < JOINT_WARMUP_EPOCHS:
            return float(epoch + 1) / float(max(1, JOINT_WARMUP_EPOCHS))

        progress = (epoch - JOINT_WARMUP_EPOCHS) / max(1, JOINT_EPOCHS - JOINT_WARMUP_EPOCHS)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    use_amp = USE_AMP and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for epoch in range(JOINT_EPOCHS):
        stage1_model.train()
        stage2_model.train()

        optimizer.zero_grad()

        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Joint Epoch {epoch + 1}/{JOINT_EPOCHS} [Train]")

        for step, (inputs, targets) in enumerate(pbar):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, enabled=use_amp):
                stage1_pred = stage1_model(inputs)
                final_pred = stage2_model(stage1_pred, inputs)

                los_mask = inputs[:, LOS_CH:LOS_CH + 1]
                edge_mask = stage2_model.edge_extractor(inputs[:, BUILDING_CH:BUILDING_CH + 1])

                loss = criterion(
                    final_pred,
                    targets,
                    los_mask=los_mask,
                    edge_mask=edge_mask,
                )

                loss = loss / ACCUM_STEPS

            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if ((step + 1) % ACCUM_STEPS == 0) or ((step + 1) == len(train_loader)):
                if use_amp:
                    scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(stage1_model.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(stage2_model.parameters(), max_norm=1.0)

                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad()

            train_loss += loss.item() * ACCUM_STEPS
            pbar.set_postfix({"loss": f"{loss.item() * ACCUM_STEPS:.5f}"})

        scheduler.step()
        train_loss /= len(train_loader)

        val_out = evaluate_combined(
            stage1_model,
            stage2_model,
            val_loader,
            criterion,
            device,
            use_amp,
        )

        print(
            f"Joint Epoch {epoch + 1} | "
            f"Train Loss: {train_loss:.5f} | "
            f"Val Loss: {val_out['loss']:.5f} | "
            f"RMSE: {val_out['rmse_db']:.3f} dB | "
            f"MAE: {val_out['mae_db']:.3f} dB | "
            f"Recall: {val_out['recall']:.3f} | "
            f"F1: {val_out['f1']:.3f}"
        )

        if val_out["rmse_db"] < best_rmse:
            best_rmse = val_out["rmse_db"]
            save_combined(BEST_COMBINED_PATH, stage1_model, stage2_model, best_rmse, epoch + 1)
            print(f"-> Saved BEST jointly fine-tuned model! RMSE: {best_rmse:.3f} dB")

    return best_rmse


def main():
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = True

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")

    if not os.path.exists(STAGE1_BEST_PATH):
        raise FileNotFoundError(
            f"Stage 1 best model not found: {STAGE1_BEST_PATH}\n"
            "Please run train_stage1.py first."
        )

    train_ds = RadioMapH5Dataset(TRAIN_H5, augment=True)
    val_ds = RadioMapH5Dataset(VAL_H5, augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    stage1_model = Stage1Model(
        input_ch=INPUT_CHANNELS,
        base=STAGE1_BASE_CHANNELS,
        fno_modes=STAGE1_FNO_MODES,
    ).to(device)

    print(f"Loading Stage 1 weights from {STAGE1_BEST_PATH}")
    stage1_state = torch.load(STAGE1_BEST_PATH, map_location=device, weights_only=True)
    stage1_model.load_state_dict(stage1_state)

    stage2_model, best_rmse = train_stage2_phase1(
        stage1_model,
        train_loader,
        val_loader,
        device,
    )

    if JOINT_FINETUNE:
        best_rmse = joint_finetune(
            stage1_model,
            stage2_model,
            train_loader,
            val_loader,
            device,
            best_rmse,
        )

    print("\nStage 2 training complete!")
    print(f"Best combined validation RMSE: {best_rmse:.3f} dB")
    print(f"Best combined checkpoint saved to: {BEST_COMBINED_PATH}")


if __name__ == "__main__":
    main()