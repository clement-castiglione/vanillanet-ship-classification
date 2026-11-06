"""VanillaNet-7 (Chen et al., NeurIPS 2023, https://arxiv.org/abs/2305.12972)
adapted to 1-channel 192x128 ship images.

This is the training-time architecture: each block still contains two 1x1
convolutions, two BatchNorms and the "deep training" activation between them.
"""

import torch
import torch.nn as nn


# Base channel counts of VanillaNet-7 in the paper: stem, then one entry per block.
BASE_STEM_CHANNELS = 512
BASE_BLOCK_CHANNELS = (1024, 1024, 2048, 4096, 4096)


class SeriesInformedActivation(nn.Module):
    """Series informed activation function (SIAF).

    Follows the paper's formula: for each channel, add a learned bias, apply
    ReLU, then take a learned weighted sum over a (2n+1)x(2n+1) neighbourhood.
    That weighted sum is exactly a depthwise convolution without bias.

    The architecture splits the work of a regular convolution in two: 1x1
    convolutions mix channels, the SIAF mixes spatial neighbours.
    """

    def __init__(self, channels, sia_n=3):
        super().__init__()
        self.series_informed_bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.series_informed_conv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=2 * sia_n + 1,
            padding="same",
            groups=channels,
            bias=False,
        )

    def forward(self, x):
        return self.series_informed_conv(torch.relu(x + self.series_informed_bias))


class VanillaNetConvBlock(nn.Module):
    """Basic VanillaNet block: conv1x1 -> BN -> act(lmbd) -> conv1x1 -> BN -> SIAF.

    During training, `lmbd` goes from 0 to 1 and turns the activation between
    the two convolutions from ReLU into the identity ("deep training"). Once
    lmbd = 1, everything before the SIAF is linear and can be fused into a
    single 1x1 convolution for inference.
    """

    def __init__(self, in_channels, out_channels, sia_n=3):
        super().__init__()
        self.first_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.first_batch_norm = nn.BatchNorm2d(out_channels)
        self.second_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.second_batch_norm = nn.BatchNorm2d(out_channels)
        self.series_informed_activation = SeriesInformedActivation(out_channels, sia_n)

    def internal_activation_1d(self, x, lmbd):
        return (1 - lmbd) * torch.relu(x) + lmbd * x

    def forward(self, x, lmbd):
        # Everything up to the SIAF becomes a single convolution at inference
        out = self.first_conv(x)
        out = self.first_batch_norm(out)
        out = self.internal_activation_1d(out, lmbd)
        out = self.second_conv(out)
        out = self.second_batch_norm(out)
        return self.series_informed_activation(out)


class VanillaNet7(nn.Module):
    """VanillaNet-7 for 1-channel images.

    The only architectural change from the paper is `in_channels=1` in the stem
    (grayscale input). `width` scales every channel count: 1.0 is the paper's
    model (used in the competition), 0.5 is the half-width variant.

    Convolutions followed by a BatchNorm have no bias: the BatchNorm subtracts
    the mean, which cancels it. The competition model kept these redundant
    biases (~25k parameters out of 69M).
    """

    def __init__(self, num_classes, width=1.0, sia_n=3):
        super().__init__()
        self.num_classes = num_classes
        self.width = width
        self.sia_n = sia_n
        stem_ch = int(BASE_STEM_CHANNELS * width)
        c1, c2, c3, c4, c5 = (int(c * width) for c in BASE_BLOCK_CHANNELS)

        self.stem = nn.Conv2d(in_channels=1, out_channels=stem_ch, kernel_size=4, stride=4, bias=False)
        self.stem_batch_norm = nn.BatchNorm2d(stem_ch)
        self.stem_series_informed_activation = SeriesInformedActivation(stem_ch, sia_n)

        self.block_1_1 = VanillaNetConvBlock(stem_ch, c1, sia_n)
        self.block_1_2 = VanillaNetConvBlock(c1, c2, sia_n)
        self.max_pooling_1 = nn.MaxPool2d(kernel_size=2)

        self.block_2 = VanillaNetConvBlock(c2, c3, sia_n)
        self.max_pooling_2 = nn.MaxPool2d(kernel_size=2)

        self.block_3 = VanillaNetConvBlock(c3, c4, sia_n)
        self.max_pooling_3 = nn.MaxPool2d(kernel_size=2)

        self.block_4 = VanillaNetConvBlock(c4, c5, sia_n)

        self.classifier_avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier_dropout = nn.Dropout(p=0.05)
        self.classifier_conv = nn.Conv2d(in_channels=c5, out_channels=num_classes, kernel_size=1)

        # Weight initialisation from the paper
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.2)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, lmbd):
        out = self.stem(x)
        out = self.stem_batch_norm(out)
        out = self.stem_series_informed_activation(out)

        out = self.block_1_1(out, lmbd)
        out = self.block_1_2(out, lmbd)
        out = self.max_pooling_1(out)

        out = self.block_2(out, lmbd)
        out = self.max_pooling_2(out)

        out = self.block_3(out, lmbd)
        out = self.max_pooling_3(out)

        out = self.block_4(out, lmbd)

        out = self.classifier_avg_pooling(out)
        out = self.classifier_dropout(out)
        out = self.classifier_conv(out)
        return torch.flatten(out, 1)



# --- Inference-time architecture -------------------------------------------
# Same network once trained with lmbd = 1 and fused (see fusion.py): no
# BatchNorm, no deep-training activation, one 1x1 convolution per block.


class DeployVanillaNetConvBlock(nn.Module):
    """Fused block: conv1x1 -> ReLU -> depthwise conv.

    The 1x1 convolution replaces conv1x1 -> BN -> conv1x1 -> BN and also carries
    the SIAF bias. ReLU followed by the depthwise convolution is what remains
    of the SIAF.
    """

    def __init__(self, in_channels, out_channels, sia_n=3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.series_informed_conv = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=2 * sia_n + 1,
            padding="same",
            groups=out_channels,
            bias=False,
        )

    def forward(self, x):
        return self.series_informed_conv(torch.relu(self.conv(x)))


class DeployVanillaNet7(nn.Module):
    """Inference-only VanillaNet-7, built from a trained model by `fusion.fuse_model`."""

    def __init__(self, num_classes, width=1.0, sia_n=3):
        super().__init__()
        stem_ch = int(BASE_STEM_CHANNELS * width)
        c1, c2, c3, c4, c5 = (int(c * width) for c in BASE_BLOCK_CHANNELS)

        # Stem: conv4x4 (BN and SIAF bias fused in) -> ReLU -> depthwise conv
        self.stem = nn.Conv2d(in_channels=1, out_channels=stem_ch, kernel_size=4, stride=4)
        self.stem_series_informed_conv = nn.Conv2d(
            in_channels=stem_ch,
            out_channels=stem_ch,
            kernel_size=2 * sia_n + 1,
            padding="same",
            groups=stem_ch,
            bias=False,
        )

        self.block_1_1 = DeployVanillaNetConvBlock(stem_ch, c1, sia_n)
        self.block_1_2 = DeployVanillaNetConvBlock(c1, c2, sia_n)
        self.max_pooling_1 = nn.MaxPool2d(kernel_size=2)

        self.block_2 = DeployVanillaNetConvBlock(c2, c3, sia_n)
        self.max_pooling_2 = nn.MaxPool2d(kernel_size=2)

        self.block_3 = DeployVanillaNetConvBlock(c3, c4, sia_n)
        self.max_pooling_3 = nn.MaxPool2d(kernel_size=2)

        self.block_4 = DeployVanillaNetConvBlock(c4, c5, sia_n)

        # Dropout is the identity at inference, so it is simply dropped
        self.classifier_avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier_conv = nn.Conv2d(in_channels=c5, out_channels=num_classes, kernel_size=1)

    def forward(self, x):
        out = self.stem_series_informed_conv(torch.relu(self.stem(x)))

        out = self.block_1_1(out)
        out = self.block_1_2(out)
        out = self.max_pooling_1(out)

        out = self.block_2(out)
        out = self.max_pooling_2(out)

        out = self.block_3(out)
        out = self.max_pooling_3(out)

        out = self.block_4(out)

        out = self.classifier_avg_pooling(out)
        out = self.classifier_conv(out)
        return torch.flatten(out, 1)
