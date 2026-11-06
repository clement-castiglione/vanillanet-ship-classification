import numpy as np
import torch

from vanillanet_ships.augment import cutmix_bbox, mix_batch, smooth_labels


def test_smoothed_labels_are_distributions():
    targets = smooth_labels(torch.tensor([0, 3, 9]), num_classes=10, smoothing=0.12)
    torch.testing.assert_close(targets.sum(dim=1), torch.ones(3))
    assert targets[0, 0] == targets.max()


def test_cutmix_bbox_stays_inside_image():
    np.random.seed(0)
    for lam in np.linspace(0.01, 0.99, 50):
        x1, y1, x2, y2 = cutmix_bbox(192, 128, lam)
        assert 0 <= x1 <= x2 <= 192
        assert 0 <= y1 <= y2 <= 128


def test_mixed_targets_stay_distributions():
    np.random.seed(0)
    torch.manual_seed(0)
    images = torch.randn(8, 1, 128, 192)
    targets = smooth_labels(torch.randint(0, 10, (8,)), num_classes=10, smoothing=0.12)
    # Repeat to go through both mixup and cutmix
    for _ in range(20):
        mixed, mixed_targets = mix_batch(images, targets, mixup_alpha=0.2, cutmix_alpha=1.0)
        assert mixed.shape == images.shape
        torch.testing.assert_close(mixed_targets.sum(dim=1), torch.ones(8))
