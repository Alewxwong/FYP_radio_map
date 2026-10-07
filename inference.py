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
# 1. CONFIGURATION
# ==========================================================
DATA_DIR = r"C:\Users\user\Desktop\Fgo\dataset\processed_data"
OUTPUT_DIR = r"C:\Users\user\Desktop\Fgo\dataset\checkpoints"

TEST_H5 = os.path.join(DATA_DIR, "radiomapseer_multifidelity_test.h5")
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "stage2_combined_best.pth")

VIS_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "inference_visualization.png")

BATCH_SIZE = 4
NUM_VIS_SAMPLES = 4

# -1 = evaluate on the FULL test set.
# Set e.g. 500 for a quick check.
MAX_TEST_SAMPLES = -1

SEED = 42

# Fallback model config if checkpoint has no config stored.
FALLBACK_STAGE1_BASE = 32
FALLBACK_STAGE1_MODES = 16
FALLBACK_STAGE2_BASE = 32
FALLBACK_STAGE2_MODES = 16

# ==========================================================
# 2. dB / SIGNAL STRENGTH CONVERSION
# ==========================================================
# RadioMapSeer maps pixel 0-255 to roughly -186 dBm ... -47 dBm
# when simulated with its reference transmitter power.
DB_MIN = -186.0
DB_RANGE = 139.0

# RadioMapSeer reference Tx power (the power used to simulate the dataset).
DATASET_TX_DBM = 23.0

# The Tx power you want to DISPLAY in the SS map.
# Change this to simulate a different base station, e.g. 30, 43, 46 dBm.
TX_POWER_DBM = 23.0

TX_GAIN_DBI = 0.0
RX_GAIN_DBI = 0.0

# Colorbar window for the SS map (dBm).
RSS_VMIN = -150.0
RSS_VMAX = -50.0

# Outage threshold in normalized scale (same as training).
OUTAGE_THRESHOLD = 0.2

# Optionally save the numeric RSS maps of the visualized samples.
SAVE_RSS_NPY = False


def norm_to_rss_dbm(pred_norm,
                    tx_power_dbm=TX_POWER_DBM,
                    tx_gain_dbi=TX_GAIN_DBI,
                    rx_gain_dbi=RX_GAIN_DBI):
    """
    Convert normalized model output [0, 1] to Received Signal Strength (dBm).

    Steps:
      1) map_dbm      = dataset-scale received power at the dataset Tx (23 dBm)
      2) path_gain_db = map_dbm - DATASET_TX_DBM   (environment-only attenuation)
      3) rss_dbm      = path_gain_db + your Tx power + antenna gains
    """
    pred_norm = np.clip(pred_norm, 0.0, 1.0)

    map_dbm = DB_MIN + pred_norm * DB_RANGE
    path_gain_db = map_dbm - DATASET_TX_DBM
    rss_dbm = path_gain_db + tx_power_dbm + tx_gain_dbi + rx_gain_dbi

    return rss_dbm


# ==========================================================
# 3. MODEL LOADING
# ==========================================================
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
        print("WARNING: no stage2_state_dict found. Using Stage 1 only.")
        stage2_model = None

    stage1_model.eval()
    if stage2_model is not None:
        stage2_model.eval()

    return stage1_model, stage2_model


# ==========================================================
# 4. TEST-SET EVALUATION
# ==========================================================
def evaluate_test(stage1_model, stage2_model, device):
    print("\nEvaluating on test set...")

    test_ds = RadioMapH5Dataset(TEST_H5, augment=False)

    if MAX_TEST_SAMPLES is not None and MAX_TEST_SAMPLES > 0 and MAX_TEST_SAMPLES < len(test_ds):
        rng = np.random.RandomState(SEED)
        indices = np.sort(rng.choice(len(test_ds), size=MAX_TEST_SAMPLES, replace=False)).tolist()
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

    return metrics_stage1.compute(), metrics_final.compute()


# ==========================================================
# 5. VISUALIZATION (with SS map, without error zoom)
# ==========================================================
def visualize(stage1_model, stage2_model, device):
    print("\nGenerating visualization...")

    with h5py.File(TEST_H5, "r") as f:
        num_samples = f["inputs"].shape[0]
        n = min(NUM_VIS_SAMPLES, num_samples)

        rng = np.random.RandomState(SEED + 1)
        indices = np.sort(rng.choice(num_samples, size=n, replace=False))

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

    # Signal strength map in dBm
    rss_dbm = norm_to_rss_dbm(final_pred)

    if SAVE_RSS_NPY:
        for i in range(n):
            np.save(
                os.path.join(OUTPUT_DIR, f"rss_map_sample{indices[i]}.npy"),
                rss_dbm[i, 0],
            )
        print("RSS .npy maps saved.")

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
        rss = rss_dbm[i, 0]
        error = np.abs(gt - fin)

        # 1. Building mask
        axes[i, 0].imshow(building, cmap="gray")
        axes[i, 0].set_title("Building Mask", fontsize=10)
        axes[i, 0].axis("off")

        # 2. Ground truth (normalized)
        im1 = axes[i, 1].imshow(gt, cmap="jet", vmin=0.0, vmax=1.0)
        axes[i, 1].set_title("Ground Truth", fontsize=10)
        axes[i, 1].axis("off")
        fig.colorbar(im1, ax=axes[i, 1], fraction=0.046, pad=0.04)

        # 3. Stage 1
        im2 = axes[i, 2].imshow(s1, cmap="jet", vmin=0.0, vmax=1.0)
        axes[i, 2].set_title("Stage 1", fontsize=10)
        axes[i, 2].axis("off")
        fig.colorbar(im2, ax=axes[i, 2], fraction=0.046, pad=0.04)

        # 4. Final prediction (normalized)
        im3 = axes[i, 3].imshow(fin, cmap="jet", vmin=0.0, vmax=1.0)
        axes[i, 3].set_title("Final Prediction", fontsize=10)
        axes[i, 3].axis("off")
        fig.colorbar(im3, ax=axes[i, 3], fraction=0.046, pad=0.04)

        # 5. Signal Strength map (dBm)
        im4 = axes[i, 4].imshow(rss, cmap="jet", vmin=RSS_VMIN, vmax=RSS_VMAX)
        axes[i, 4].set_title(f"SS Map (dBm) @ Tx={TX_POWER_DBM:.0f} dBm", fontsize=10)
        axes[i, 4].axis("off")
        cb4 = fig.colorbar(im4, ax=axes[i, 4], fraction=0.046, pad=0.04)
        cb4.set_label("dBm", fontsize=9)

        # 6. Absolute error (normalized scale)
        im5 = axes[i, 5].imshow(error, cmap="hot", vmin=0.0, vmax=0.2)
        axes[i, 5].set_title("Absolute Error", fontsize=10)
        axes[i, 5].axis("off")
        fig.colorbar(im5, ax=axes[i, 5], fraction=0.046, pad=0.04)

    plt.savefig(VIS_OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Visualization saved to: {VIS_OUTPUT_PATH}")

    return rss_dbm


# ==========================================================
# 6. MAIN
# ==========================================================
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

    # Physical interpretation of the outage threshold
    outage_dbm = (
        DB_MIN + OUTAGE_THRESHOLD * DB_RANGE - DATASET_TX_DBM + TX_POWER_DBM
    )
    print("\nSignal strength info:")
    print(f"  Dataset reference Tx power : {DATASET_TX_DBM:.1f} dBm")
    print(f"  Display Tx power           : {TX_POWER_DBM:.1f} dBm")
    print(f"  Outage threshold (0.2 norm): {outage_dbm:.1f} dBm at display Tx power")

    print("=" * 70)

    visualize(stage1_model, stage2_model, device)


if __name__ == "__main__":
    main()