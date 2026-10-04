import math
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset


# ==========================================================
# Constants matching your data_process.py
# ==========================================================
BUILDING_CH = 0
ANT_CH = 1
SDF_CH = 2
DIST_CH = 3
LOS_CH = 4

DB_RANGE = 139.0
PIXEL_MAX = 255.0
OUTAGE_THRESHOLD = 0.2


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gn(channels: int):
    """
    GroupNorm helper.
    Uses 8 groups when possible, otherwise falls back to 1 group.
    """
    groups = 8 if channels % 8 == 0 else 1
    return nn.GroupNorm(groups, channels)


# ==========================================================
# Dataset
# ==========================================================
class RadioMapH5Dataset(Dataset):
    def __init__(self, h5_path, augment=False):
        self.h5_path = h5_path
        self.augment = augment
        self._file = None

    @property
    def file(self):
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r")
        return self._file

    def __len__(self):
        return self.file["inputs"].shape[0]

    def __getitem__(self, idx):
        inputs = self.file["inputs"][idx].astype(np.float32)
        target = self.file["irt2_targets"][idx].astype(np.float32) / PIXEL_MAX

        # Safety normalization.
        inputs = np.clip(inputs, 0.0, 1.0)
        target = np.clip(target, 0.0, 1.0)

        inputs = torch.from_numpy(inputs).float()
        target = torch.from_numpy(target).float()

        if self.augment:
            inputs, target = augment_sample(inputs, target)

        return inputs, target


def augment_sample(inputs, target):
    """
    inputs: C x H x W
    target: 1 x H x W

    Geometric augmentations are safe here because all input channels
    are spatially aligned with the target map.
    """
    # Horizontal flip
    if torch.rand(1).item() < 0.5:
        inputs = torch.flip(inputs, dims=[2])
        target = torch.flip(target, dims=[2])

    # Vertical flip
    if torch.rand(1).item() < 0.5:
        inputs = torch.flip(inputs, dims=[1])
        target = torch.flip(target, dims=[1])

    # Random 90-degree rotation
    k = torch.randint(0, 4, (1,)).item()
    if k > 0:
        inputs = torch.rot90(inputs, k, dims=[1, 2])
        target = torch.rot90(target, k, dims=[1, 2])

    return inputs, target


# ==========================================================
# Metrics
# ==========================================================
class MetricAccumulator:
    """
    Computes global RMSE/MAE in dB and outage precision/recall/F1.
    This is more stable than averaging per-batch metrics.
    """

    def __init__(self, threshold=OUTAGE_THRESHOLD):
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.se = 0.0
        self.ae = 0.0
        self.count = 0
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0

    @torch.no_grad()
    def update(self, pred, target):
        pred = pred.detach().float()
        target = target.detach().float()

        diff = pred - target

        self.se += torch.sum(diff * diff).item()
        self.ae += torch.sum(torch.abs(diff)).item()
        self.count += target.numel()

        pred_outage = pred < self.threshold
        target_outage = target < self.threshold

        self.tp += torch.sum(pred_outage & target_outage).item()
        self.fp += torch.sum(pred_outage & (~target_outage)).item()
        self.fn += torch.sum((~pred_outage) & target_outage).item()

    def compute(self):
        rmse_norm = math.sqrt(self.se / max(1, self.count))
        mae_norm = self.ae / max(1, self.count)

        rmse_db = rmse_norm * DB_RANGE
        mae_db = mae_norm * DB_RANGE

        precision = self.tp / (self.tp + self.fp + 1e-8)
        recall = self.tp / (self.tp + self.fn + 1e-8)
        f1 = 2.0 * precision * recall / (precision + recall + 1e-8)

        return {
            "rmse_db": rmse_db,
            "mae_db": mae_db,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }


# ==========================================================
# Edge extractor
# ==========================================================
class EdgeExtractor(nn.Module):
    """
    Extracts building-boundary edge map from the building mask.
    This is more reliable than using the current unsigned SDF only.
    """

    def __init__(self):
        super().__init__()
        kx = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=torch.float32
        ).view(1, 1, 3, 3)

        ky = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            dtype=torch.float32
        ).view(1, 1, 3, 3)

        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)

    def forward(self, building_mask):
        gx = F.conv2d(building_mask, self.kx, padding=1)
        gy = F.conv2d(building_mask, self.ky, padding=1)
        edge = torch.sqrt(gx.pow(2) + gy.pow(2) + 1e-8)
        edge = torch.clamp(edge, 0.0, 1.0)
        return edge


# ==========================================================
# FNO building blocks
# ==========================================================
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1.0 / (in_channels * out_channels)

        self.weights1 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    @staticmethod
    def compl_mul2d(x, weights):
        return torch.einsum("bixy,ioxy->boxy", x, weights)

    def forward(self, x):
        batch_size, _, h, w = x.shape
        x_ft = torch.fft.rfft2(x)

        out_ft = torch.zeros(
            batch_size,
            self.out_channels,
            h,
            w // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )

        # Robustness if input size is smaller than expected.
        modes1 = min(self.modes1, h // 2)
        modes2 = min(self.modes2, w // 2 + 1)

        if modes1 <= 0 or modes2 <= 0:
            return x

        w1 = self.weights1[:, :, :modes1, :modes2]
        w2 = self.weights2[:, :, :modes1, :modes2]

        out_ft[:, :, :modes1, :modes2] = self.compl_mul2d(
            x_ft[:, :, :modes1, :modes2], w1
        )

        out_ft[:, :, -modes1:, :modes2] = self.compl_mul2d(
            x_ft[:, :, -modes1:, :modes2], w2
        )

        return torch.fft.irfft2(out_ft, s=(h, w))


class FNOBlockV2(nn.Module):
    """
    Improved FNO block:
    spectral conv + 1x1 local conv + normalization + GELU + residual.
    """

    def __init__(self, channels, modes):
        super().__init__()
        self.spectral = SpectralConv2d(channels, channels, modes, modes)
        self.local = nn.Conv2d(channels, channels, kernel_size=1)
        self.norm = gn(channels)
        self.act = nn.GELU()

    def forward(self, x):
        y = self.spectral(x) + self.local(x)
        y = self.norm(y)
        y = self.act(y)
        return x + y


# ==========================================================
# CNN blocks
# ==========================================================
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            gn(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            gn(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            gn(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            gn(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.down(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class DilatedBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2, bias=False),
            gn(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=4, dilation=4, bias=False),
            gn(channels),
            nn.GELU(),
        )

    def forward(self, x):
        return x + self.block(x)


# ==========================================================
# Stage 1: Physics-guided U-FNO
# ==========================================================
class Stage1Model(nn.Module):
    """
    Stage 1 model:
    - Uses transmitter distance map to create a physics baseline.
    - Predicts a residual correction.
    - U-shaped CNN skips preserve sharp local details.
    - FNO bottleneck captures global spectral propagation.
    """

    def __init__(self, input_ch=5, base=32, fno_modes=16):
        super().__init__()

        # Learnable decay constant for the distance-based baseline.
        self.tau = nn.Parameter(torch.tensor(0.25))

        c0 = base
        c1 = base * 2
        c2 = base * 3
        c3 = min(base * 4, 128)

        # +1 because we concatenate the physics baseline channel.
        self.stem = ConvBlock(input_ch + 1, c0)

        self.down1 = DownBlock(c0, c1)
        self.down2 = DownBlock(c1, c2)
        self.down3 = DownBlock(c2, c3)

        self.fno = nn.Sequential(
            FNOBlockV2(c3, fno_modes),
            FNOBlockV2(c3, fno_modes),
            FNOBlockV2(c3, fno_modes),
        )

        self.up3 = UpBlock(c3 + c2, c2)
        self.up2 = UpBlock(c2 + c1, c1)
        self.up1 = UpBlock(c1 + c0, c0)

        head_mid = max(c0 // 2, 16)
        self.head = nn.Sequential(
            nn.Conv2d(c0, head_mid, kernel_size=3, padding=1, bias=False),
            gn(head_mid),
            nn.GELU(),
            nn.Conv2d(head_mid, 1, kernel_size=1),
        )

        self._zero_init_head()

    def _zero_init_head(self):
        last_conv = self.head[-1]
        nn.init.zeros_(last_conv.weight)
        nn.init.zeros_(last_conv.bias)

    def forward(self, x):
        # x: B, 5, H, W
        tx_dist = x[:, DIST_CH:DIST_CH + 1]

        tau = torch.clamp(self.tau, min=0.05)
        baseline = torch.exp(-tx_dist / tau)
        baseline = torch.clamp(baseline, 0.0, 1.0)

        # Provide the baseline explicitly as an extra input channel.
        xin = torch.cat([x, baseline], dim=1)

        x0 = self.stem(xin)          # full resolution
        x1 = self.down1(x0)          # 1/2
        x2 = self.down2(x1)          # 1/4
        x3 = self.down3(x2)          # 1/8

        x3 = self.fno(x3)

        u = self.up3(x3, x2)
        u = self.up2(u, x1)
        u = self.up1(u, x0)

        residual = self.head(u)
        pred = baseline + residual
        pred = torch.clamp(pred, 0.0, 1.0)

        return pred


# ==========================================================
# Stage 2: Diffraction-aware residual refinement
# ==========================================================
class Stage2Model(nn.Module):
    """
    Stage 2 model:
    - Takes Stage 1 prediction + all five geometry/input channels.
    - Adds explicit building-edge map.
    - Uses dilated convs + FNO blocks + U-shaped skips.
    - Predicts a residual correction.
    """

    def __init__(self, input_ch=5, base=32, fno_modes=16):
        super().__init__()

        self.edge_extractor = EdgeExtractor()

        # Input channels:
        # 1: stage1 prediction
        # input_ch: original inputs
        # 1: building edge map
        total_in = 1 + input_ch + 1

        c0 = base
        c1 = base * 2
        c2 = min(base * 4, 128)

        self.stem = ConvBlock(total_in, c0)

        self.down1 = DownBlock(c0, c1)
        self.down2 = DownBlock(c1, c2)

        self.mid = nn.Sequential(
            DilatedBlock(c2),
            FNOBlockV2(c2, fno_modes),
            FNOBlockV2(c2, fno_modes),
        )

        self.up2 = UpBlock(c2 + c1, c1)
        self.up1 = UpBlock(c1 + c0, c0)

        head_mid = max(c0 // 2, 16)
        self.head = nn.Sequential(
            nn.Conv2d(c0, head_mid, kernel_size=3, padding=1, bias=False),
            gn(head_mid),
            nn.GELU(),
            nn.Conv2d(head_mid, 1, kernel_size=1),
        )

        self._zero_init_head()

    def _zero_init_head(self):
        last_conv = self.head[-1]
        nn.init.zeros_(last_conv.weight)
        nn.init.zeros_(last_conv.bias)

    def forward(self, stage1_pred, inputs):
        # stage1_pred: B, 1, H, W
        # inputs: B, 5, H, W

        stage1_pred = torch.clamp(stage1_pred, 0.0, 1.0)

        building = inputs[:, BUILDING_CH:BUILDING_CH + 1]
        edge = self.edge_extractor(building)

        x = torch.cat([stage1_pred, inputs, edge], dim=1)

        s0 = self.stem(x)
        s1 = self.down1(s0)
        s2 = self.down2(s1)

        m = self.mid(s2)

        u = self.up2(m, s1)
        u = self.up1(u, s0)

        residual = self.head(u)

        final = stage1_pred + residual
        final = torch.clamp(final, 0.0, 1.0)

        return final


# ==========================================================
# Loss functions
# ==========================================================
class SobolevLoss(nn.Module):
    def __init__(self):
        super().__init__()

        kx = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=torch.float32
        ).view(1, 1, 3, 3)

        ky = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            dtype=torch.float32
        ).view(1, 1, 3, 3)

        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)

    def forward(self, pred, target):
        pred_dx = F.conv2d(pred, self.kx, padding=1)
        pred_dy = F.conv2d(pred, self.ky, padding=1)

        target_dx = F.conv2d(target, self.kx, padding=1)
        target_dy = F.conv2d(target, self.ky, padding=1)

        return F.mse_loss(pred_dx, target_dx) + F.mse_loss(pred_dy, target_dy)


class FFTLoss(nn.Module):
    def forward(self, pred, target):
        pred_fft = torch.fft.rfft2(pred)
        target_fft = torch.fft.rfft2(target)
        return F.l1_loss(pred_fft.abs(), target_fft.abs())


class RadioMapLoss(nn.Module):
    """
    Composite loss optimized for both RMSE and outage/edge behavior.

    If you only care about RMSE, reduce outage_weight/shadow_weight/edge_weight.
    If you need better outage F1, increase outage_weight slightly.
    """

    def __init__(
        self,
        l1_weight=1.0,
        mse_weight=0.5,
        sobolev_weight=0.10,
        fft_weight=0.02,
        outage_weight=0.5,
        shadow_weight=0.5,
        edge_weight=0.5,
        outage_threshold=OUTAGE_THRESHOLD,
    ):
        super().__init__()
        self.l1_weight = l1_weight
        self.mse_weight = mse_weight
        self.sobolev_weight = sobolev_weight
        self.fft_weight = fft_weight
        self.outage_weight = outage_weight
        self.shadow_weight = shadow_weight
        self.edge_weight = edge_weight
        self.outage_threshold = outage_threshold

        self.sobolev = SobolevLoss()
        self.fft = FFTLoss()

    def forward(self, pred, target, los_mask=None, edge_mask=None):
        pred = torch.clamp(pred, 0.0, 1.0)
        target = torch.clamp(target, 0.0, 1.0)

        weight = torch.ones_like(target)

        # Give a mild boost to low-signal/outage pixels.
        outage_mask = (target < self.outage_threshold).float()
        weight = weight + self.outage_weight * outage_mask

        # Give a mild boost to non-LoS shadow regions.
        if los_mask is not None:
            los_mask = torch.clamp(los_mask, 0.0, 1.0)
            weight = weight + self.shadow_weight * (1.0 - los_mask)

        # Give a mild boost to building boundaries.
        if edge_mask is not None:
            edge_mask = torch.clamp(edge_mask, 0.0, 1.0)
            weight = weight + self.edge_weight * edge_mask

        l1_loss = torch.mean(weight * torch.abs(pred - target))
        mse_loss = torch.mean(weight * (pred - target).pow(2))

        sob_loss = self.sobolev(pred, target)
        fft_loss = self.fft(pred, target)

        total = (
            self.l1_weight * l1_loss
            + self.mse_weight * mse_loss
            + self.sobolev_weight * sob_loss
            + self.fft_weight * fft_loss
        )

        return total