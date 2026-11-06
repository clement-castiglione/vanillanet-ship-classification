import torch
import torch.nn as nn
import torch.nn.functional as F

from vanillanet_ships import DeployVanillaNet7, VanillaNet7
from vanillanet_ships.fusion import fuse_1x1_convs, fuse_conv_bn, fuse_model


def randomize_bn(bn):
    """Give a BatchNorm non-trivial statistics and affine parameters."""
    with torch.no_grad():
        bn.running_mean.uniform_(-1, 1)
        bn.running_var.uniform_(0.5, 2)
        bn.weight.uniform_(0.5, 1.5)
        bn.bias.uniform_(-1, 1)


def test_fuse_conv_bn():
    torch.manual_seed(0)
    conv = nn.Conv2d(8, 16, kernel_size=4, stride=4)
    bn = nn.BatchNorm2d(16)
    randomize_bn(bn)
    conv.eval(), bn.eval()

    x = torch.randn(2, 8, 32, 48)
    weight, bias = fuse_conv_bn(conv, bn)
    with torch.no_grad():
        expected = bn(conv(x))
        fused = F.conv2d(x, weight, bias, stride=conv.stride, padding=conv.padding)

    torch.testing.assert_close(fused, expected, atol=1e-5, rtol=1e-5)


def test_fuse_1x1_convs():
    torch.manual_seed(0)
    first = nn.Conv2d(8, 16, kernel_size=1)
    second = nn.Conv2d(16, 32, kernel_size=1)

    x = torch.randn(2, 8, 6, 9)
    weight, bias = fuse_1x1_convs(first.weight, first.bias, second.weight, second.bias)
    with torch.no_grad():
        expected = second(first(x))
        fused = F.conv2d(x, weight, bias)

    assert weight.shape == (32, 8, 1, 1)
    torch.testing.assert_close(fused, expected, atol=1e-5, rtol=1e-5)


def randomize_trained_state(model):
    """Mimic a trained model: BatchNorm statistics and SIAF biases are non-zero."""
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            randomize_bn(m)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("series_informed_bias"):
                p.uniform_(-0.5, 0.5)


def test_fuse_model_matches_trained_model():
    torch.manual_seed(0)
    # A small width keeps the test fast; fusion does not depend on width
    model = VanillaNet7(num_classes=10, width=0.25)
    randomize_trained_state(model)
    model.eval()

    deploy = fuse_model(model)
    assert isinstance(deploy, DeployVanillaNet7)

    x = torch.randn(2, 1, 128, 192)
    with torch.no_grad():
        expected = model(x, lmbd=1.0)
        fused = deploy(x)

    # Relative tolerance: activations grow through the network and float32
    # rounding differs between the fused and unfused computations
    torch.testing.assert_close(fused, expected, rtol=1e-4, atol=1e-4 * expected.abs().max().item())
