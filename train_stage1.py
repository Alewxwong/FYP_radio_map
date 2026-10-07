import os
import math
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import (
    get_device,
    RadioMapH5Dataset,
    Stage1Model,
    RadioMapLoss,
    EdgeExtractor,
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

BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "stage1_fno_best.pth")
LAST_CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "stage1_last.pth")

INPUT_CHANNELS = 5

# If you have enough VRAM, try BASE_CHANNELS=48.
# If OOM, reduce to 24 or 16.
BASE_CHANNELS = 32
FNO_MODES = 16

BATCH_SIZE = 4
ACCUM_STEPS = 4  # effective batch size = BATCH_SIZE * ACCUM_STEPS
EPOCHS = 80
LEARNING_RATE = 6e-4
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 5

NUM_WORKERS = 0
USE_AMP = False  # Keep False for stable FFT training. Set True only if you test carefully.

RESUME = True

# Loss weights.
L1_WEIGHT = 1.0
MSE_WEIGHT = 0.5
SOBOLEV_WEIGHT = 0.10
FFT_WEIGHT = 0.02
OUTAGE_WEIGHT = 0.5
SHADOW_WEIGHT = 0.5
EDGE_WEIGHT = 0.5


def evaluate_stage1(model, val_loader, criterion, edge_extractor, device, use_amp):
    model.eval()
    metrics = MetricAccumulator()

    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, enabled=use_amp):
                preds = model(inputs)

                los_mask = inputs[:, LOS_CH:LOS_CH + 1]
                edge_mask = edge_extractor(inputs[:, BUILDING_CH:BUILDING_CH + 1])

                loss = criterion(
                    preds,
                    targets,
                    los_mask=los_mask,
                    edge_mask=edge_mask,
                )

            total_loss += loss.item()
            num_batches += 1
            metrics.update(preds, targets)

    out = metrics.compute()
    out["loss"] = total_loss / max(1, num_batches)
    return out


def main():
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = True

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")

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

    model = Stage1Model(
        input_ch=INPUT_CHANNELS,
        base=BASE_CHANNELS,
        fno_modes=FNO_MODES,
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

    edge_extractor = EdgeExtractor().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            return float(epoch + 1) / float(max(1, WARMUP_EPOCHS))

        progress = (epoch - WARMUP_EPOCHS) / max(1, EPOCHS - WARMUP_EPOCHS)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    use_amp = USE_AMP and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    start_epoch = 0
    best_rmse = float("inf")

    if RESUME and os.path.exists(LAST_CHECKPOINT_PATH):
        print(f"[RESUME] Loading checkpoint from {LAST_CHECKPOINT_PATH}")
        ckpt = torch.load(LAST_CHECKPOINT_PATH, map_location=device, weights_only=False)

        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        start_epoch = ckpt["epoch"]
        best_rmse = ckpt.get("best_rmse", float("inf"))

        if use_amp and "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])

        print(f"[RESUME] Starting from epoch {start_epoch + 1}, best RMSE = {best_rmse:.4f} dB")

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        optimizer.zero_grad()

        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Train]")

        for step, (inputs, targets) in enumerate(pbar):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, enabled=use_amp):
                preds = model(inputs)

                los_mask = inputs[:, LOS_CH:LOS_CH + 1]
                edge_mask = edge_extractor(inputs[:, BUILDING_CH:BUILDING_CH + 1])

                loss = criterion(
                    preds,
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

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

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

        val_out = evaluate_stage1(
            model,
            val_loader,
            criterion,
            edge_extractor,
            device,
            use_amp,
        )

        print(
            f"Epoch {epoch + 1} | "
            f"Train Loss: {train_loss:.5f} | "
            f"Val Loss: {val_out['loss']:.5f} | "
            f"RMSE: {val_out['rmse_db']:.3f} dB | "
            f"MAE: {val_out['mae_db']:.3f} dB | "
            f"Recall: {val_out['recall']:.3f} | "
            f"F1: {val_out['f1']:.3f}"
        )

        is_best = val_out["rmse_db"] < best_rmse

        if is_best:
            best_rmse = val_out["rmse_db"]
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"-> Saved BEST Stage 1 model! RMSE: {best_rmse:.3f} dB")

        save_dict = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_rmse": best_rmse,
        }

        if use_amp:
            save_dict["scaler_state_dict"] = scaler.state_dict()

        torch.save(save_dict, LAST_CHECKPOINT_PATH)

    print("\nStage 1 training complete!")
    print(f"Best validation RMSE: {best_rmse:.3f} dB")


if __name__ == "__main__":
    main()