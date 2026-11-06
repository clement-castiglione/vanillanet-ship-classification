"""Label smoothing and batch-level mixup / cutmix.

Both augmentations blend two images of the batch and blend their targets with
the same proportion, which is why the loss works on soft targets (BCE) rather
than on class indices.
"""

import numpy as np
import torch
import torch.nn.functional as F


def smooth_labels(labels, num_classes, smoothing):
    one_hot = F.one_hot(labels, num_classes=num_classes).float()
    return one_hot * (1.0 - smoothing) + smoothing / num_classes


def cutmix_bbox(width, height, lam):
    """Random box covering about (1 - lam) of the image, clipped to its borders."""
    cut_ratio = np.sqrt(1.0 - lam)
    cut_w = max(1, int(width * cut_ratio))
    cut_h = max(1, int(height * cut_ratio))
    cx = np.random.randint(width)
    cy = np.random.randint(height)
    x1 = np.clip(cx - cut_w // 2, 0, width)
    y1 = np.clip(cy - cut_h // 2, 0, height)
    x2 = np.clip(cx + cut_w // 2, 0, width)
    y2 = np.clip(cy + cut_h // 2, 0, height)
    return x1, y1, x2, y2


def mix_batch(images, targets, mixup_alpha, cutmix_alpha):
    """Apply mixup or cutmix (50/50) to a batch and mix the soft targets accordingly.

    Mixup overlays two images with transparency; cutmix pastes a random patch
    of one image onto another.
    """
    index = torch.randperm(images.size(0), device=images.device)
    if np.random.rand() < 0.5:
        lam = np.random.beta(mixup_alpha, mixup_alpha) if mixup_alpha > 0 else 1.0
        mixed = lam * images + (1 - lam) * images[index]
    else:
        lam = np.random.beta(cutmix_alpha, cutmix_alpha) if cutmix_alpha > 0 else 1.0
        _, _, height, width = images.shape
        x1, y1, x2, y2 = cutmix_bbox(width, height, lam)
        mixed = images.clone()
        mixed[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]
        # The box may have been clipped: use the area actually pasted
        lam = 1.0 - (x2 - x1) * (y2 - y1) / float(width * height)
    return mixed, lam * targets + (1 - lam) * targets[index]
