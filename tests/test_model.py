import pytest
import torch

from vanillanet_ships import VanillaNet7


@pytest.mark.parametrize("width", [1.0, 0.5])
def test_output_shape(width):
    model = VanillaNet7(num_classes=10, width=width).eval()
    x = torch.randn(2, 1, 128, 192)
    with torch.inference_mode():
        out = model(x, lmbd=1.0)
    assert out.shape == (2, 10)


def test_half_width_has_fewer_parameters():
    full = sum(p.numel() for p in VanillaNet7(num_classes=10).parameters())
    half = sum(p.numel() for p in VanillaNet7(num_classes=10, width=0.5).parameters())
    # 1x1 convolutions dominate and scale with width^2
    assert 3.5 < full / half < 4.0
