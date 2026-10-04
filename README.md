Below is a complete updated `README.md` you can use for the new version of your FYP codebase.

```md
# Real-Time 2D Radio Environment Mapping using Physics-Unrolled Neural Operators

**Final Year Project (FYP)**  
**Group Members:** Wong Tsz Hin Alex, Tan King Ho Tony  
**Date:** October 2026  
**Version:** 2.0

---

## 1. Project Overview

Radio Environment Maps (REMs) are spatial representations of wireless signal strength across a geographic area. They are essential for 5G/B5G network planning, access-point placement, coverage optimization, and outage-zone prediction.

Traditionally, generating high-fidelity radio maps requires computationally expensive ray-tracing simulations, such as Sionna RT or WinProp. These simulators are accurate but too slow for real-time interactive network planning. Standard deep learning models, such as U-Net, can predict radio maps quickly, but they often over-smooth sharp signal transitions near building corners, shadow boundaries, and diffraction edges.

### 🚀 Our Solution

This project implements a **two-stage physics-guided neural operator framework** for fast and accurate outdoor radio map prediction.

Instead of treating radio maps as generic images, the model explicitly uses physically meaningful priors:

- building geometry,
- transmitter location,
- transmitter distance map,
- signed distance field around buildings,
- line-of-sight / shadow mask.

The current Version 2.0 architecture uses:

1. **Stage 1 — Physics-Guided U-FNO**
   - Predicts the broad radio coverage map.
   - Uses a transmitter-distance-based physics baseline.
   - Uses a U-shaped CNN encoder/decoder with an FNO bottleneck.
   - Learns a residual correction on top of the baseline.

2. **Stage 2 — Edge-Aware Diffraction Refinement Network**
   - Refines the Stage 1 prediction near building edges and shadow boundaries.
   - Uses an explicit building-edge map extracted from the building mask.
   - Uses dilated convolutions, FNO blocks, and U-shaped skip connections.
   - Predicts a residual correction that is added to the Stage 1 output.

The final model predicts high-fidelity signal strength maps in milliseconds while preserving sharp shadow transitions better than a plain shallow FNO model.

---

## 2. What Is New in Version 2.0

This version includes major improvements over the original training pipeline.

### ✅ Architecture Improvements

| Component | Old Version | New Version |
|---|---|---|
| Stage 1 | Shallow FNO + local conv branch | Physics-guided U-FNO with CNN skips |
| Stage 2 | Dilated CNN with SDF gate | Edge-aware residual U-FNO refinement |
| Output | Unbounded prediction | Clamped to `[0, 1]` |
| Physics prior | Implicit only | Explicit distance-based baseline |
| Edge information | SDF only | Explicit Sobel building-edge map |
| Model selection | Mainly outage F1 | Validation RMSE in dB |
| Metrics | Per-batch averaging | Global pixel-level accumulation |

### ✅ Dataset Improvements

| Component | Old Version | New Version |
|---|---|---|
| SDF | Unsigned distance outside buildings | True signed distance field remapped to `[0, 1]` |
| LoS | Exact ray-casting | Exact ray-casting, regenerated cleanly |
| Target | IRT2 raw PNG values | IRT2 stored as raw `0-255` float maps |
| Additional target | IRT4 optional | IRT4 stored for future high-fidelity fine-tuning |
| Splits | Sometimes only test processed | Train / val / test all generated |

### ✅ Training Improvements

- Global RMSE/MAE computation over the whole validation set.
- Model checkpoint selection based on **validation RMSE in dB**.
- Gradient accumulation for larger effective batch size.
- Warmup + cosine learning-rate schedule.
- Gradient clipping for stable training.
- Optional joint fine-tuning of Stage 1 and Stage 2.
- Fixed LoS channel usage bug from previous Stage 2 code.

---

## 3. Methodology

### 3.1 Input Features

The model receives five geometry-prior channels:

```text
Channel 0: Building mask
Channel 1: Antenna heatmap
Channel 2: Signed distance field
Channel 3: Transmitter distance map
Channel 4: Line-of-sight / shadow mask
```

All input channels are normalized or clipped to `[0, 1]`.

#### Building Mask

Binary map of urban structures:

```text
1 = building
0 = free space
```

#### Antenna Heatmap

A Gaussian blob indicating the transmitter location.

#### Signed Distance Field

Version 2.0 uses a true signed distance field:

```text
sdf = distance_outside_buildings - distance_inside_buildings
```

It is normalized and remapped to `[0, 1]`:

```text
0.0 ≈ deep inside building
0.5 ≈ near building boundary
1.0 ≈ far outside building
```

This is more informative than the old unsigned SDF.

#### Transmitter Distance Map

Normalized Euclidean distance from the transmitter to every pixel.

This channel provides the baseline path-loss scaffold.

#### LoS / Shadow Mask

Binary ray-casted mask:

```text
1 = line-of-sight from transmitter
0 = shadowed by building
```

---

## 4. Model Architecture

### 4.1 Stage 1: Physics-Guided U-FNO

Stage 1 predicts the main radio map.

It first creates a physics-like baseline from the transmitter distance map:

```text
baseline = exp(-tx_distance / tau)
```

Then the network predicts a residual correction:

```text
Stage1_output = clamp(baseline + learned_residual, 0, 1)
```

This helps the model avoid learning the obvious distance-based path-loss decay from scratch.

#### Stage 1 Structure

```text
Input:
    5 original channels + 1 baseline channel = 6 channels
    shape: B x 6 x 256 x 256

Stem ConvBlock:
    B x 32 x 256 x 256

DownBlock 1:
    B x 64 x 128 x 128

DownBlock 2:
    B x 96 x 64 x 64

DownBlock 3:
    B x 128 x 32 x 32

FNO Bottleneck:
    3 x FNOBlockV2
    B x 128 x 32 x 32

UpBlock 3 + skip:
    B x 96 x 64 x 64

UpBlock 2 + skip:
    B x 64 x 128 x 128

UpBlock 1 + skip:
    B x 32 x 256 x 256

Head Conv:
    B x 1 x 256 x 256

Output:
    clamp(baseline + residual, 0, 1)
```

Default Stage 1 settings:

```python
BASE_CHANNELS = 32
FNO_MODES = 16
```

---

### 4.2 Stage 2: Edge-Aware Diffraction Refinement

Stage 2 refines the Stage 1 prediction.

It receives:

```text
Stage 1 prediction:      1 channel
Original inputs:         5 channels
Building edge map:       1 channel
--------------------------------
Total input:             7 channels
```

The building-edge map is computed using a Sobel filter on the building mask.

#### Stage 2 Structure

```text
Input:
    B x 7 x 256 x 256

Stem ConvBlock:
    B x 32 x 256 x 256

DownBlock 1:
    B x 64 x 128 x 128

DownBlock 2:
    B x 128 x 64 x 64

Middle Block:
    DilatedBlock
    FNOBlockV2
    FNOBlockV2
    B x 128 x 64 x 64

UpBlock 2 + skip:
    B x 64 x 128 x 128

UpBlock 1 + skip:
    B x 32 x 256 x 256

Head Conv:
    B x 1 x 256 x 256

Output:
    clamp(Stage1_prediction + residual, 0, 1)
```

Default Stage 2 settings:

```python
STAGE2_BASE_CHANNELS = 32
STAGE2_FNO_MODES = 16
```

---

## 5. Loss Functions

The new training pipeline uses a composite loss designed to improve both pixel accuracy and edge preservation.

### 5.1 L1 Loss

Encourages robust pixel-level accuracy.

### 5.2 MSE Loss

Directly penalizes squared error, which is useful for improving RMSE.

### 5.3 Sobolev Gradient Loss

Penalizes mismatches in spatial gradients:

```text
∇prediction ≈ ∇target
```

This helps preserve sharp shadow boundaries.

### 5.4 FFT Loss

Encourages agreement in the frequency domain:

```text
|FFT(prediction)| ≈ |FFT(target)|
```

This helps preserve global structure and periodic spatial patterns.

### 5.5 Weighted Shadow / Edge / Outage Loss

The loss can optionally give extra weight to:

- non-line-of-sight regions,
- building-edge regions,
- low-signal outage regions.

Default loss weights:

```python
L1_WEIGHT = 1.0
MSE_WEIGHT = 0.5
SOBOLEV_WEIGHT = 0.10
FFT_WEIGHT = 0.02
OUTAGE_WEIGHT = 0.5
SHADOW_WEIGHT = 0.5
EDGE_WEIGHT = 0.5
```

If the main objective is RMSE, keep these weights moderate.  
If outage F1 becomes too low, increase `OUTAGE_WEIGHT`, `SHADOW_WEIGHT`, or `EDGE_WEIGHT` slightly.

---

## 6. Dataset

The project uses the **RadioMapSeer** dataset.

It contains:

```text
700 urban maps
80 transmitter locations per map
56,000 total samples
```

### 6.1 Data Split

The split is based on map ID:

```text
Train: map_id < 560
Val:   560 <= map_id < 630
Test:  map_id >= 630
```

Approximate split sizes:

```text
Train: 44,800 samples
Val:    5,600 samples
Test:   5,600 samples
```

### 6.2 Target Labels

The current stable training pipeline uses:

```text
IRT2 targets
```

IRT4 targets are also stored in the HDF5 files for future high-fidelity fine-tuning.

Target normalization during training:

```python
target = irt2_target / 255.0
```

The raw PNG target range `0-255` corresponds approximately to a 139 dB dynamic range.

Therefore:

```python
DB_RANGE = 139.0
rmse_db = rmse_normalized * 139.0
```

---

## 7. Data Preprocessing

The preprocessing pipeline is implemented in:

```text
data_process.py
```

It generates:

```text
radiomapseer_multifidelity_train.h5
radiomapseer_multifidelity_val.h5
radiomapseer_multifidelity_test.h5
```

Each HDF5 file contains:

```text
inputs        shape: N x 5 x 256 x 256
irt2_targets  shape: N x 1 x 256 x 256
irt4_targets  shape: N x 1 x 256 x 256
map_ids       shape: N
tx_ids        shape: N
```

### 7.1 Important Preprocessing Notes

#### Exact LoS generation is slow

The LoS mask is computed using exact pixel-by-pixel ray casting.

For each sample:

```text
256 x 256 = 65,536 rays
```

Therefore, full dataset generation may take:

```text
30 minutes to several hours
```

depending on CPU speed and number of workers.

#### Signed SDF is stored in `[0, 1]`

The SDF is internally signed, but it is remapped to `[0, 1]` so that it remains compatible with the training pipeline.

---

## 8. Project Structure

```text
Fgo/
├── dataset/
│   ├── RadioMapSeer/
│   │   ├── png/
│   │   │   ├── buildings_complete/
│   │   │   └── antennas/
│   │   ├── gain/
│   │   │   ├── IRT2/
│   │   │   └── IRT4/
│   │   ├── dataset.csv
│   │   └── processed_data/
│   │       ├── radiomapseer_multifidelity_train.h5
│   │       ├── radiomapseer_multifidelity_val.h5
│   │       └── radiomapseer_multifidelity_test.h5
│   └── checkpoints/
│       ├── stage1_fno_best.pth
│       ├── stage1_last.pth
│       ├── stage2_combined_best.pth
│       ├── stage2_last.pth
│       └── inference_visualization.png
│
├── common.py
├── data_process.py
├── train_stage1.py
├── train_stage2.py
├── inference.py
├── environment.yml
└── README.md
```

---

## 9. File Descriptions

### `data_process.py`

Generates the `.h5` dataset files.

Responsibilities:

- parse `dataset.csv`,
- load building masks,
- load antenna masks,
- load IRT2 and IRT4 target maps,
- compute signed distance field,
- compute transmitter distance map,
- compute antenna heatmap,
- compute exact LoS mask,
- write train/val/test HDF5 files.

Run:

```bash
python data_process.py
```

---

### `common.py`

Contains shared code used by training and inference.

Includes:

- dataset class,
- metric accumulator,
- model blocks,
- Stage 1 model,
- Stage 2 model,
- loss functions,
- edge extractor.

---

### `train_stage1.py`

Trains the Stage 1 physics-guided U-FNO.

Run:

```bash
python train_stage1.py
```

Outputs:

```text
stage1_fno_best.pth
stage1_last.pth
```

---

### `train_stage2.py`

Trains the Stage 2 edge-aware refinement network.

Training has two phases:

#### Phase 1

Stage 1 is frozen.  
Stage 2 is trained to refine Stage 1 predictions.

#### Phase 2

Optional joint fine-tuning:

```text
Stage 1 learning rate: small
Stage 2 learning rate: larger
```

Run:

```bash
python train_stage2.py
```

Outputs:

```text
stage2_combined_best.pth
stage2_last.pth
```

---

### `inference.py`

Runs inference using the best combined Stage 1 + Stage 2 checkpoint.

Run:

```bash
python inference.py
```

Outputs:

```text
Test-set metrics
inference_visualization.png
```

---

## 10. Installation

The project uses Conda.

### Prerequisites

```text
Python 3.10+
NVIDIA GPU recommended
CUDA-compatible PyTorch installation
```

### Setup

Create the environment:

```bash
conda env create -f environment.yml
```

Activate:

```bash
conda activate fyp_radio_map
```

---

## 11. How to Run

### Step 1: Generate Dataset

```bash
python data_process.py
```

This creates:

```text
processed_data/radiomapseer_multifidelity_train.h5
processed_data/radiomapseer_multifidelity_val.h5
processed_data/radiomapseer_multifidelity_test.h5
```

⚠️ This step may take a long time because exact LoS masks are generated.

---

### Step 2: Train Stage 1

```bash
python train_stage1.py
```

Stage 1 trains the physics-guided U-FNO.

The best model is selected using validation RMSE in dB.

---

### Step 3: Train Stage 2

```bash
python train_stage2.py
```

Stage 2 first trains with Stage 1 frozen.

Then it optionally performs joint fine-tuning of Stage 1 and Stage 2.

The best combined checkpoint is saved as:

```text
stage2_combined_best.pth
```

---

### Step 4: Inference

```bash
python inference.py
```

This evaluates the model on the test set and saves visualization results.

---

## 12. Default Training Configuration

### Stage 1

```python
BASE_CHANNELS = 32
FNO_MODES = 16
BATCH_SIZE = 4
ACCUM_STEPS = 4
EPOCHS = 80
LEARNING_RATE = 6e-4
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 5
```

Effective batch size:

```text
4 x 4 = 16
```

---

### Stage 2 Phase 1

```python
STAGE2_BASE_CHANNELS = 32
STAGE2_FNO_MODES = 16
BATCH_SIZE = 4
ACCUM_STEPS = 4
STAGE2_EPOCHS = 40
STAGE2_LR = 3e-4
```

---

### Stage 2 Phase 2: Joint Fine-Tuning

```python
JOINT_FINETUNE = True
JOINT_EPOCHS = 10
JOINT_STAGE1_LR = 1e-5
JOINT_STAGE2_LR = 5e-5
```

---

## 13. Evaluation Metrics

The project tracks both image-quality metrics and wireless-deployment metrics.

### 13.1 RMSE in dB

Root mean square error converted to decibels:

```text
rmse_db = rmse_normalized * 139.0
```

This is the main model-selection metric in Version 2.0.

---

### 13.2 MAE in dB

Mean absolute error converted to decibels:

```text
mae_db = mae_normalized * 139.0
```

---

### 13.3 Outage Metrics

Outage regions are defined as:

```text
normalized signal < 0.2
```

The model tracks:

```text
Precision
Recall
Outage F1
```

These are important because a model can have a low RMSE while still missing sharp coverage holes.

---

## 14. Important Implementation Details

### 14.1 Predictions are clamped

Both Stage 1 and Stage 2 outputs are clamped:

```python
prediction = torch.clamp(prediction, 0.0, 1.0)
```

This prevents invalid values outside the normalized signal range.

---

### 14.2 Stage heads are zero-initialized

The final convolution layers in Stage 1 and Stage 2 are initialized with zeros.

This means:

- Stage 1 initially predicts the physics baseline.
- Stage 2 initially predicts an identity refinement.

This improves training stability.

---

### 14.3 Global metric accumulation

Validation and test metrics are computed globally over all pixels.

This is more reliable than averaging RMSE per mini-batch.

---

### 14.4 Correct LoS channel usage

The previous Stage 2 implementation incorrectly used transmitter distance as the LoS channel.

Version 2.0 fixes this:

```python
LOS_CH = 4
los_mask = inputs[:, LOS_CH:LOS_CH + 1]
```

---

## 15. Recommended Experiment Workflow

For a clean experiment, use the following order:

```bash
python data_process.py
python train_stage1.py
python train_stage2.py
python inference.py
```

If you change the dataset preprocessing, train from scratch.

Do not resume old checkpoints if:

- the input channels changed,
- the SDF representation changed,
- the model architecture changed,
- the target normalization changed.

---

## 16. If You Run Out of GPU Memory

Reduce model size:

```python
BASE_CHANNELS = 24
STAGE1_BASE_CHANNELS = 24
STAGE2_BASE_CHANNELS = 24
FNO_MODES = 12
STAGE1_FNO_MODES = 12
STAGE2_FNO_MODES = 12
BATCH_SIZE = 2
ACCUM_STEPS = 8
```

If you have more GPU memory, you can increase:

```python
BASE_CHANNELS = 48
STAGE1_BASE_CHANNELS = 48
STAGE2_BASE_CHANNELS = 48
FNO_MODES = 24
STAGE1_FNO_MODES = 24
STAGE2_FNO_MODES = 24
```

Increase gradually and monitor GPU memory usage.

---

## 17. Troubleshooting

### Problem: `FileNotFoundError: radiomapseer_multifidelity_train.h5`

Solution:

Run data processing first:

```bash
python data_process.py
```

Make sure `DATA_DIR` points to the correct processed folder.

---

### Problem: Data processing is very slow

Cause:

Exact LoS generation is CPU-heavy.

Solution:

Let it run in the background.  
The time required depends strongly on CPU core count.

---

### Problem: Training runs out of memory

Solution:

Reduce:

```python
BASE_CHANNELS
FNO_MODES
BATCH_SIZE
```

Increase:

```python
ACCUM_STEPS
```

---

### Problem: RMSE improves but outage F1 drops

Solution:

Increase outage/shadow weighting slightly:

```python
OUTAGE_WEIGHT = 1.0
SHADOW_WEIGHT = 0.75
EDGE_WEIGHT = 0.75
```

---

### Problem: Outage F1 improves but RMSE becomes worse

Solution:

Reduce edge/outage weighting:

```python
OUTAGE_WEIGHT = 0.2
SHADOW_WEIGHT = 0.2
EDGE_WEIGHT = 0.2
```

---

## 18. Current Limitations

The current Version 2.0 pipeline is stable and strong, but there are still possible extensions.

### Possible Future Improvements

1. **True Graph Neural Operator Stage 2**
   - Build graphs over building-edge pixels.
   - Perform message passing along diffraction edges.

2. **Wavelet Branch**
   - Add wavelet-domain processing to better capture localized high-frequency details.

3. **IRT4 Fine-Tuning**
   - Use stored IRT4 targets for final high-fidelity fine-tuning.

4. **Patch-Based Training**
   - Train on random crops to increase effective dataset diversity.

5. **Test-Time Augmentation**
   - Use flips/rotations during inference and average predictions.

6. **Larger FNO Modes**
   - Increase spectral modes if GPU memory allows.

---

## 19. References

This project builds upon and adapts the following research:

1. R. Levie, Ç. Yapar, G. Kutyniok, and G. Caire,  
   “RadioUNet: Fast radio map estimation with convolutional neural networks,”  
   *IEEE Transactions on Wireless Communications*, vol. 20, no. 6, pp. 4001–4015, Jun. 2021.

2. J.-H. Lee, O. G. Serbetci, D. Panneer Selvam, and A. F. Molisch,  
   “PMNet: Robust pathloss map prediction via supervised learning,”  
   in *Proc. IEEE Global Communications Conference (GLOBECOM)*, Kuala Lumpur, Malaysia, 2023, pp. 4601–4606.

3. Y. Li, Z. Li, Z. Gao, and T. Chen,  
   “Geo2SigMap: High-fidelity RF signal mapping using geographic databases,”  
   *arXiv preprint arXiv:2312.14303*, 2023.

4. R. U. Murshed, S. U. Rahman, M. Tang, and E. Soltanaghai,  
   “Physics-unrolled neural operator for wireless field modeling,”  
   *arXiv preprint arXiv:2608.18495*, 2026.

5. Z. Li, N. Kovachki, K. Azizzadenesheli, B. Liu, K. Bhattacharya, A. Stuart, and A. Anandkumar,  
   “Fourier Neural Operator for Parametric Partial Differential Equations,”  
   *International Conference on Learning Representations (ICLR)*, 2021.
```