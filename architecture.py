import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def gn(num_channels):
    for groups in (16, 8, 4, 2, 1):
        if num_channels % groups == 0:
            return nn.GroupNorm(groups, num_channels)
    return nn.GroupNorm(1, num_channels)


class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.norm1 = gn(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

        self.norm2 = gn(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

        if in_channels != out_channels or stride != 1:
            self.skip = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False) #convolution to match dimensions of input and output when they differ in channels or spatial dimensions
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = self.conv1(self.relu1(self.norm1(x)))
        out = self.conv2(self.relu2(self.norm2(out)))
        return out + identity


def conv_block(in_channels, out_channels, kernel_size=3, padding=1):
    """Compatibility helper now backed by a residual 3D block."""
    return ResidualBlock3D(in_channels, out_channels, stride=1)


class SpatialFiLMLayer(nn.Module):
    def __init__(self, num_channels, hidden_ratio=0.25):
        super().__init__()
        hidden = max(1, int(num_channels * hidden_ratio))

        self.trunk = nn.Sequential(
            nn.Conv3d(num_channels, hidden, kernel_size=3, padding=1, bias=False),
            gn(hidden),
            nn.ReLU(inplace=True),
        )

        self.gamma_conv = nn.Conv3d(hidden, num_channels, kernel_size=1, bias=True)
        self.beta_conv = nn.Conv3d(hidden, num_channels, kernel_size=1, bias=True)

        # Identity-init FiLM: gamma≈0 and beta≈0 at startup.
        nn.init.zeros_(self.gamma_conv.weight)
        nn.init.zeros_(self.gamma_conv.bias)
        nn.init.zeros_(self.beta_conv.weight)
        nn.init.zeros_(self.beta_conv.bias)

    def forward(self, z_qsm, z_flair):
        h = self.trunk(z_flair)
        gamma = self.gamma_conv(h)
        beta = self.beta_conv(h)
        return (1.0 + gamma) * z_qsm + beta


class FiLMLayer(nn.Module):
    def __init__(self, num_channels):
        super(FiLMLayer, self).__init__()
        self.spatial_film = SpatialFiLMLayer(num_channels)

    def forward(self, z_qsm, z_flair):
        return self.spatial_film(z_qsm, z_flair)


class ClassificationHead(nn.Module):
    """Classification head for binary Rim/NoRim prediction"""
    def __init__(self, in_channels, num_classes=1, dropout=0.3):
        super(ClassificationHead, self).__init__()
        self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, max(32, in_channels // 2)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(max(32, in_channels // 2), num_classes)
        )

    def forward(self, x):
        x = self.gap(x)
        x = self.classifier(x)
        return x

    def init_logit_bias(self, pos_prior):
        pi = min(max(float(pos_prior), 1e-6), 1.0 - 1e-6)
        bias_value = math.log(pi / (1.0 - pi))
        last_linear = self.classifier[-1]
        with torch.no_grad():
            last_linear.bias.fill_(bias_value)


class SEBlock3D(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc1 = nn.Conv3d(channels, hidden, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv3d(hidden, channels, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        s = self.pool(x)
        s = self.fc2(self.relu(self.fc1(s)))
        return x * self.sigmoid(s)


class MultiModalClassifier(nn.Module):
    """
    FiLM-based multi-modal classifier for FLAIR + QSM.
    
    Architecture:
        - QSM encoder (backbone branch - primary signal)
        - FLAIR encoder (conditioning branch - context)
        - FiLM modulation layer
        - Classification head
    
    Input order:
        inputs[0] = FLAIR (B, 1, D, H, W)
        inputs[1] = QSM   (B, 1, D, H, W)
    """
    def __init__(self, in_channels_list, base_filters=16, num_classes=1, dropout=0.3, modality_names=None, pos_prior=None):
        super(MultiModalClassifier, self).__init__()

        c1 = min(base_filters, 256)
        c2 = min(base_filters * 2, 256)
        c3 = min(base_filters * 4, 256)
        c4 = min(base_filters * 8, 256)

        self._input_modalities = len(in_channels_list)
        self.flair_index = 0
        self.qsm_index = 1 if len(in_channels_list) > 1 else 0
        if modality_names is not None:
            lower = [m.lower() for m in modality_names]
            if 'flair' in lower:
                self.flair_index = lower.index('flair')
            if 'qsm' in lower:
                self.qsm_index = lower.index('qsm')

        flair_in = in_channels_list[self.flair_index] if self.flair_index < len(in_channels_list) else 1
        qsm_in = in_channels_list[self.qsm_index] if self.qsm_index < len(in_channels_list) else in_channels_list[0]

        # QSM main stream
        self.qsm_stem = nn.Conv3d(qsm_in, c1, kernel_size=3, padding=1, bias=False)
        self.qsm_stage1a = ResidualBlock3D(c1, c1, stride=1)
        self.qsm_stage1b = ResidualBlock3D(c1, c1, stride=1)
        self.qsm_stage2a = ResidualBlock3D(c1, c2, stride=(2, 2, 1))  
        self.qsm_stage2b = ResidualBlock3D(c2, c2, stride=1)
        self.qsm_stage3a = ResidualBlock3D(c2, c3, stride=(2, 2, 2))
        self.qsm_stage3b = ResidualBlock3D(c3, c3, stride=1)
        self.qsm_stage4a = ResidualBlock3D(c3, c4, stride=(2, 2, 2))
        self.qsm_stage4b = ResidualBlock3D(c4, c4, stride=1)

        # FLAIR conditioning stream
        self.flair_stem = nn.Conv3d(flair_in, c1, kernel_size=3, padding=1, bias=False)
        self.flair_stage1a = ResidualBlock3D(c1, c1, stride=1)
        self.flair_stage1b = ResidualBlock3D(c1, c1, stride=1)
        self.flair_stage2a = ResidualBlock3D(c1, c2, stride=(2, 2, 1))
        self.flair_stage2b = ResidualBlock3D(c2, c2, stride=1)

        # Spatial FiLM at intermediate resolution
        self.film_mid = SpatialFiLMLayer(c2)
        self.fuse_conv = nn.Sequential(
            nn.Conv3d(2 * c2, c2, kernel_size=1, bias=False),
            gn(c2),
            nn.ReLU(inplace=True),
        )

        self.se3 = SEBlock3D(c3)
        self.se4 = SEBlock3D(c4)

        self.classifier = ClassificationHead(c4, num_classes=num_classes, dropout=dropout)
        if pos_prior is not None:
            self.classifier.init_logit_bias(pos_prior)

    def _split_modalities(self, inputs):
        if len(inputs) == 1:
            return None, inputs[0]
        if self.flair_index < len(inputs) and self.qsm_index < len(inputs):
            if self.flair_index != self.qsm_index:
                return inputs[self.flair_index], inputs[self.qsm_index]
        return inputs[0], inputs[1]
    
    def forward(self, inputs, return_features=False):
        """
        Args:
            inputs: List of tensors [FLAIR, QSM]
                - inputs[0]: FLAIR (B, 1, D, H, W)
                - inputs[1]: QSM   (B, 1, D, H, W)
            return_features: Return fused embeddings if True
        
        Returns:
            If return_features=False:
                logits: (B, 1) - raw classification scores
            If return_features=True:
                (logits, z_fused): tuple where z_fused is (B, C) fused embedding before classifier
        """
        flair, qsm = self._split_modalities(inputs)

        z_qsm = self.qsm_stage1a(self.qsm_stem(qsm))
        z_qsm = self.qsm_stage1b(z_qsm)
        z_qsm = self.qsm_stage2a(z_qsm)
        z_qsm = self.qsm_stage2b(z_qsm)

        if flair is not None:
            z_flair = self.flair_stage1a(self.flair_stem(flair))
            z_flair = self.flair_stage1b(z_flair)
            z_flair = self.flair_stage2a(z_flair)
            z_flair = self.flair_stage2b(z_flair)
            z_qsm = self.film_mid(z_qsm, z_flair)
            z_qsm = self.fuse_conv(torch.cat([z_qsm, z_flair], dim=1))

        z_qsm = self.qsm_stage3a(z_qsm)
        z_qsm = self.qsm_stage3b(z_qsm)
        z_qsm = self.se3(z_qsm)
        z_qsm = self.qsm_stage4a(z_qsm)
        z_qsm = self.qsm_stage4b(z_qsm)
        z_qsm = self.se4(z_qsm)

        if return_features:
            z_fused = F.adaptive_avg_pool3d(z_qsm, (1, 1, 1)).view(z_qsm.size(0), -1)

        logits = self.classifier(z_qsm)

        if return_features:
            return logits, z_fused
        return logits

