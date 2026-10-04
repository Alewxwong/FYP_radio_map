import os
import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Subset

from common import (
    get_device,
    RadioMapH5Dataset,
    Stage1Model,
    Stage2Model,
    MetricAccumulator,
)


# ==========================================================
# Configuration
# ==========================================================
DATA_DIR = r"C:\Users\user\Desktop\Fgo\dataset\processed_data"
OUTPUT_DIR = r"C:\Users\user\Desktop\Fgo\dataset\checkpoints"

TEST_H5 = os.path.join(DATA_DIR, "radiomapseer_multifidelity_test.h5")
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "stage2_combined_best.pth")

VIS_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "inference_visualization.png")

BATCH_SIZE = 4
NUM_VIS_SAMPLES = 4

# Set to -1 to evaluate on the full test set.
# Set to e.g. 500 if you want a quick test.
MAX_TEST_SAMPLES = -1

SEED = 42

# Fallback model config if checkpoint does not contain config.
FALLBACK_STAGE1_BASE = 32
FALLBACK_STAGE1_MODES = 16
FALLBACK_STAGE2_BASE = 32
FALLBACK_STAGE2_MODES = 16


def load_models(device):
    print(f"Loading checkpoint: {CHECKPOINT_PATH}")

    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)

    cfg = ckpt.get("config", {})

    stage1_base = cfg.get("stage1_base", FALLBACK_STAGE1_BASE)
    stage1_modes = cfg.get("stage1_modes", FALLBACK_STAGE1_MODES)
    stage2_base = cfg.get("stage2_base", FALLBACK_STAGE2_BASE)
    stage2_modes = cfg.get("stage2_modes", FALLBACK_STAGE2_MODES)
    input_ch = cfg.get("input_ch", 5)

    stage1_model = Stage1Model(
        input_ch=input_ch,
        base=stage1_base,
        fno_modes=stage1_modes,
    ).to(device)

    stage2_model = Stage2Model(
        input_ch=input_ch,
        base=stage2_base,
        fno_modes=stage2_modes,
    ).to(device)

    if "stage1_state_dict" in ckpt:
        stage1_model.load_state_dict(ckpt["stage1_state_dict"])
    else:
        raise KeyError("Checkpoint does not contain 'stage1_state_dict'.")

    if "stage2_state_dict" in ckpt:
        stage2_model.load_state_dict(ckpt["stage2_state_dict"])
    else:
        print("WARNING: checkpoint does not contain stage2_state_dict. Using Stage 1 only.")
        stage2_model = None

    stage1_model.eval()
    if stage2_model is not None:
        stage2_model.eval()

    return stage1_model, stage2_model


def evaluate_test(stage1_model, stage2_model, device):
    print("\nEvaluating on test set...")

    test_ds = RadioMapH5Dataset(TEST_H5, augment=False)

    if MAX_TEST_SAMPLES is not None and MAX_TEST_SAMPLES > 0 and MAX_TEST_SAMPLES < len(test_ds):
        rng = np.random.RandomState(SEED)
        indices = rng.choice(len(test_ds), size=MAX_TEST_SAMPLES, replace=False)
        indices = np.sort(indices).tolist()
        test_ds = Subset(test_ds, indices)
        print(f"Evaluating on {len(test_ds)} randomly selected test samples.")
    else:
        print(f"Evaluating on all {len(test_ds)} test samples.")

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    metrics_stage1 = MetricAccumulator()
    metrics_final = MetricAccumulator()

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            stage1_pred = stage1_model(inputs)

            if stage2_model is not None:
                final_pred = stage2_model(stage1_pred, inputs)
            else:
                final_pred = stage1_pred

            metrics_stage1.update(stage1_pred, targets)
            metrics_final.update(final_pred, targets)

    stage1_out = metrics_stage1.compute()
    final_out = metrics_final.compute()

    return stage1_out, final_out


def visualize(stage1_model, stage2_model, device):
    print("\nGenerating visualization...")

    with h5py.File(TEST_H5, "r") as f:
        num_samples = f["inputs"].shape[0]
        n = min(NUM_VIS_SAMPLES, num_samples)

        rng = np.random.RandomState(SEED + 1)
        indices = rng.choice(num_samples, size=n, replace=False)
        indices = np.sort(indices)

        inputs = torch.tensor(f["inputs"][indices], dtype=torch.float32)
        targets = torch.tensor(f["irt2_targets"][indices] / 255.0, dtype=torch.float32)

    inputs = inputs.to(device)
    targets = targets.to(device)

    with torch.no_grad():
        stage1_pred = stage1_model(inputs)

        if stage2_model is not None:
            final_pred = stage2_model(stage1_pred, inputs)
        else:
            final_pred = stage1_pred

    inputs = inputs.cpu().numpy()
    targets = targets.cpu().numpy()
    stage1_pred = stage1_pred.cpu().numpy()
    final_pred = final_pred.cpu().numpy()

    plt.switch_backend("Agg")

    fig, axes = plt.subplots(n, 6, figsize=(26, 4.2 * n))
    plt.subplots_adjust(wspace=0.15, hspace=0.25)

    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(n):
        building = inputs[i, 0]
        gt = targets[i, 0]
        s1 = stage1_pred[i, 0]
        fin = final_pred[i, 0]

        error = np.abs(gt - fin)

        axes[i, 0].imshow(building, cmap="gray")
        axes[i, 0].set_title("Building Mask", fontsize=10)
        axes[i, 0].axis("off")

        im1 = axes[i, 1].imshow(gt, cmap="jet", vmin=0.0, vmax=1.0)
        axes[i, 1].set_title("Ground Truth", fontsize=10)
        axes[i, 1].axis("off")
        fig.colorbar(im1, ax=axes[i, 1], fraction=0.046, pad=0.04)

        im2 = axes[i, 2].imshow(s1, cmap="jet", vmin=0.0, vmax=1.0)
        axes[i, 2].set_title("Stage 1", fontsize=10)
        axes[i, 2].axis("off")
        fig.colorbar(im2, ax=axes[i, 2], fraction=0.046, pad=0.04)

        im3 = axes[i, 3].imshow(fin, cmap="jet", vmin=0.0, vmax=1.0)
        axes[i, 3].set_title("Final Prediction", fontsize=10)
        axes[i, 3].axis("off")
        fig.colorbar(im3, ax=axes[i, 3], fraction=0.046, pad=0.04)

        im4 = axes[i, 4].imshow(error, cmap="hot", vmin=0.0, vmax=0.2)
        axes[i, 4].set_title("Absolute Error", fontsize=10)
        axes[i, 4].axis("off")
        fig.colorbar(im4, ax=axes[i, 4], fraction=0.046, pad=0.04)

        h, w = error.shape
        crop = min(64, h, w)
        im5 = axes[i, 5].imshow(error[h - crop:h, w - crop:w], cmap="hot", vmin=0.0, vmax=0.2)
        axes[i, 5].set_title("Error Zoom", fontsize=10)
        axes[i, 5].axis("off")
        fig.colorbar(im5, ax=axes[i, 5], fraction=0.046, pad=0.04)

    plt.savefig(VIS_OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Visualization saved to: {VIS_OUTPUT_PATH}")


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")

    stage1_model, stage2_model = load_models(device)

    stage1_metrics, final_metrics = evaluate_test(stage1_model, stage2_model, device)

    print("\n" + "=" * 70)
    print("TEST SET RESULTS")
    print("=" * 70)

    print("\nStage 1 only:")
    print(f"  RMSE     : {stage1_metrics['rmse_db']:.3f} dB")
    print(f"  MAE      : {stage1_metrics['mae_db']:.3f} dB")
    print(f"  Precision: {stage1_metrics['precision']:.3f}")
    print(f"  Recall   : {stage1_metrics['recall']:.3f}")
    print(f"  Outage F1: {stage1_metrics['f1']:.3f}")

    print("\nFinal Stage 1 + Stage 2:")
    print(f"  RMSE     : {final_metrics['rmse_db']:.3f} dB")
    print(f"  MAE      : {final_metrics['mae_db']:.3f} dB")
    print(f"  Precision: {final_metrics['precision']:.3f}")
    print(f"  Recall   : {final_metrics['recall']:.3f}")
    print(f"  Outage F1: {final_metrics['f1']:.3f}")

    print("\n" + "=" * 70)

    visualize(stage1_model, stage2_model, device)


if __name__ == "__main__":
    main()