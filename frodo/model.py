"""Two-stream QSM/FLAIR classifier with spatial FiLM fusion (FRODO).

Architecture only: encoders, FiLM fusion, SE blocks, classification head,
and the optional EMA weight-averaging wrapper. No optimizer or data loading.
"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


def group_norm(num_channels):
    for groups in (16, 8, 4, 2, 1):
        if num_channels % groups == 0:
            return nn.GroupNorm(groups, num_channels)
    return nn.GroupNorm(1, num_channels)


class ResidualBlock3D(nn.Module):
    """Pre-activation residual block: GN-ReLU-Conv, twice, plus a skip
    connection (1x1x1 projection when channels or stride change)."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.norm1 = group_norm(in_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm2 = group_norm(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

        if in_channels != out_channels or stride != 1:
            self.skip = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = self.conv1(F.relu(self.norm1(x), inplace=True))
        out = self.conv2(F.relu(self.norm2(out), inplace=True))
        return out + identity


class SpatialFiLMLayer(nn.Module):
    """gamma/beta generated from the FLAIR features via two 1x1x1 convs;
    modulates the QSM features as z~Q = (1 + gamma) * zQ + beta."""

    def __init__(self, num_channels):
        super().__init__()
        self.gamma_conv = nn.Conv3d(num_channels, num_channels, kernel_size=1)
        self.beta_conv = nn.Conv3d(num_channels, num_channels, kernel_size=1)
        # Identity init: gamma=0, beta=0 at the start of training.
        nn.init.zeros_(self.gamma_conv.weight)
        nn.init.zeros_(self.gamma_conv.bias)
        nn.init.zeros_(self.beta_conv.weight)
        nn.init.zeros_(self.beta_conv.bias)

    def forward(self, z_qsm, z_flair):
        gamma = self.gamma_conv(z_flair)
        beta = self.beta_conv(z_flair)
        return (1.0 + gamma) * z_qsm + beta


class SEBlock3D(nn.Module):
    """Squeeze-and-excitation: GAP -> 1x1 reduce -> ReLU -> 1x1 restore -> sigmoid -> scale."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Conv3d(channels, hidden, kernel_size=1)
        self.fc2 = nn.Conv3d(hidden, channels, kernel_size=1)

    def forward(self, x):
        s = self.pool(x)
        s = F.relu(self.fc1(s), inplace=True)
        s = torch.sigmoid(self.fc2(s))
        return x * s


class ClassificationHead(nn.Module):
    """GAP -> Linear -> ReLU -> Dropout -> Linear. Returns logits (sigmoid is
    applied via BCEWithLogitsLoss during training/inference, not here), and
    optionally the pre-final-linear embedding used by the contrastive loss."""

    def __init__(self, in_channels, num_classes=1, dropout=0.3):
        super().__init__()
        hidden = max(32, in_channels // 2)
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.pre = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.out = nn.Linear(hidden, num_classes)

    def forward(self, x, return_embedding=False):
        embedding = self.pre(self.gap(x))
        logits = self.out(embedding)
        if return_embedding:
            return logits, embedding
        return logits


def _make_stem(in_channels, c1, c2):
    return nn.ModuleDict({
        "stem": nn.Conv3d(in_channels, c1, kernel_size=3, padding=1, bias=False),
        "block1a": ResidualBlock3D(c1, c1, stride=1),
        "block1b": ResidualBlock3D(c1, c1, stride=1),
        "block2a": ResidualBlock3D(c1, c2, stride=2),
        "block2b": ResidualBlock3D(c2, c2, stride=1),
    })


def _run_stem(stem, x):
    x = stem["stem"](x)
    x = stem["block1a"](x)
    x = stem["block1b"](x)
    x = stem["block2a"](x)
    x = stem["block2b"](x)
    return x


class MultiModalClassifier(nn.Module):
    """Two-stream QSM/FLAIR classifier: symmetric encoder stems, spatial FiLM
    fusion at mid-resolution, deeper QSM-only stages with SE blocks, and a
    binary classification head."""

    def __init__(self, base_filters=16, num_classes=1, dropout=0.3):
        super().__init__()
        c1, c2, c3, c4 = base_filters, base_filters * 2, base_filters * 4, base_filters * 8

        self.qsm_stem = _make_stem(1, c1, c2)
        self.flair_stem = _make_stem(1, c1, c2)

        self.film = SpatialFiLMLayer(c2)
        self.fuse_conv = nn.Conv3d(2 * c2, c2, kernel_size=1)

        self.block3a = ResidualBlock3D(c2, c3, stride=2)
        self.block3b = ResidualBlock3D(c3, c3, stride=1)
        self.se3 = SEBlock3D(c3)
        self.block4a = ResidualBlock3D(c3, c4, stride=2)
        self.block4b = ResidualBlock3D(c4, c4, stride=1)
        self.se4 = SEBlock3D(c4)

        self.classifier = ClassificationHead(c4, num_classes=num_classes, dropout=dropout)

    def encode(self, qsm, flair):
        """Bottleneck features (pooled to vectors), before FiLM fusion.
        Used by train_ssl.py for the cross-modal regression objective."""
        z_qsm = _run_stem(self.qsm_stem, qsm)
        z_flair = _run_stem(self.flair_stem, flair)
        z_qsm_vec = F.adaptive_avg_pool3d(z_qsm, 1).flatten(1)
        z_flair_vec = F.adaptive_avg_pool3d(z_flair, 1).flatten(1)
        return z_qsm_vec, z_flair_vec

    def forward(self, qsm, flair, return_embedding=False):
        z_qsm = _run_stem(self.qsm_stem, qsm)
        z_flair = _run_stem(self.flair_stem, flair)

        z_qsm = self.film(z_qsm, z_flair)
        z_qsm = self.fuse_conv(torch.cat([z_qsm, z_flair], dim=1))

        z_qsm = self.se3(self.block3b(self.block3a(z_qsm)))
        z_qsm = self.se4(self.block4b(self.block4a(z_qsm)))

        return self.classifier(z_qsm, return_embedding=return_embedding)


class ModelEMA:
    """Optional exponential moving average of model weights (--use_ema).
    Not part of the paper; provided as an optional training utility."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.ema_model = copy.deepcopy(model).eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        msd = model.state_dict()
        for k, v in self.ema_model.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(msd[k].detach(), alpha=1.0 - self.decay)
            else:
                v.copy_(msd[k])
