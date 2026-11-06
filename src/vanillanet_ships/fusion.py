"""Inference-time fusion: turns a trained VanillaNet7 into a DeployVanillaNet7.

Valid only when the model is in eval mode (BatchNorm uses its running
statistics) and was trained until lmbd = 1 (the activation between the two 1x1
convolutions of each block is the identity). Each block then goes from
conv1x1 -> BN -> conv1x1 -> BN -> SIAF to conv1x1 -> ReLU -> depthwise conv.
"""

import torch
import torch.nn as nn

from .model import DeployVanillaNet7, DeployVanillaNetConvBlock, VanillaNet7, VanillaNetConvBlock


@torch.no_grad()
def fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d):
    """Return (weight, bias) of a single convolution equivalent to bn(conv(x)).

    W' = W * gamma / sigma
    b' = (b - mu) * gamma / sigma + beta,   with sigma = sqrt(running_var + eps)
    """
    scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    weight = conv.weight * scale.reshape(-1, 1, 1, 1)
    bias = conv.bias if conv.bias is not None else torch.zeros_like(bn.running_mean)
    return weight, (bias - bn.running_mean) * scale + bn.bias


@torch.no_grad()
def fuse_1x1_convs(w1: torch.Tensor, b1: torch.Tensor, w2: torch.Tensor, b2: torch.Tensor):
    """Return (weight, bias) of a single 1x1 convolution equivalent to two in a row.

    A 1x1 convolution is the same linear map applied to every pixel, so
    W2 (W1 x + b1) + b2 = (W2 W1) x + (W2 b1 + b2).
    """
    w1, w2 = w1[:, :, 0, 0], w2[:, :, 0, 0]
    return (w2 @ w1)[:, :, None, None], w2 @ b1 + b2


def siaf_bias(siaf) -> torch.Tensor:
    """SIAF bias as a convolution bias: (1, C, 1, 1) -> (C,).

    The SIAF computes relu(x + bias), so its bias can be added to the bias of
    the convolution that produces x.
    """
    return siaf.series_informed_bias[0, :, 0, 0]


@torch.no_grad()
def fuse_block(source: VanillaNetConvBlock, target: DeployVanillaNetConvBlock):
    """Fill `target` with the fused weights of the trained block `source`."""
    w1, b1 = fuse_conv_bn(source.first_conv, source.first_batch_norm)
    w2, b2 = fuse_conv_bn(source.second_conv, source.second_batch_norm)
    weight, bias = fuse_1x1_convs(w1, b1, w2, b2)

    target.conv.weight.copy_(weight)
    target.conv.bias.copy_(bias + siaf_bias(source.series_informed_activation))
    target.series_informed_conv.weight.copy_(source.series_informed_activation.series_informed_conv.weight)


@torch.no_grad()
def fuse_model(model: VanillaNet7) -> DeployVanillaNet7:
    """Build the inference-only model equivalent to `model` (eval mode, lmbd = 1)."""
    fused = DeployVanillaNet7(num_classes=model.num_classes, width=model.width, sia_n=model.sia_n)

    # Stem: conv4x4 -> BN -> SIAF  becomes  conv4x4 -> ReLU -> depthwise conv
    weight, bias = fuse_conv_bn(model.stem, model.stem_batch_norm)
    fused.stem.weight.copy_(weight)
    fused.stem.bias.copy_(bias + siaf_bias(model.stem_series_informed_activation))
    fused.stem_series_informed_conv.weight.copy_(model.stem_series_informed_activation.series_informed_conv.weight)

    for name in ("block_1_1", "block_1_2", "block_2", "block_3", "block_4"):
        fuse_block(getattr(model, name), getattr(fused, name))

    # Nothing to fuse in the classifier (dropout is the identity at inference)
    fused.classifier_conv.load_state_dict(model.classifier_conv.state_dict())

    return fused.eval()
