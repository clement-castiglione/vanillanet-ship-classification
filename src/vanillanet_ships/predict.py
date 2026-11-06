"""Inference with test-time augmentation (TTA) and optional checkpoint ensembling.

Each image is classified from 10 views (5 crops x 2 horizontal flips) and the
logits of all views are summed. Checkpoints are fused before inference by
default: the predictions are the same and the model is smaller and faster
(see scripts/benchmark.py).
"""

from functools import partial

import torch
from PIL import Image
from torchvision import transforms

from .data import IMG_HEIGHT, IMG_WIDTH, MEAN, STD
from .fusion import fuse_model
from .model import VanillaNet7


def build_tta_transforms(crop_ratio=0.9):
    """10 views: 4 corner crops and a center crop, each with and without flip, resized back."""
    crop_h, crop_w = int(IMG_HEIGHT * crop_ratio), int(IMG_WIDTH * crop_ratio)
    pad_h, pad_w = IMG_HEIGHT - crop_h, IMG_WIDTH - crop_w
    offsets = [(0, 0), (0, pad_w), (pad_h, 0), (pad_h, pad_w), (pad_h // 2, pad_w // 2)]

    views = []
    for top, left in offsets:
        for flip in (False, True):
            ops = [
                transforms.Lambda(lambda img, t=top, l=left: img.crop((l, t, l + crop_w, t + crop_h))),
                transforms.Resize((IMG_HEIGHT, IMG_WIDTH), interpolation=transforms.InterpolationMode.BICUBIC),
            ]
            if flip:
                ops.append(transforms.RandomHorizontalFlip(p=1.0))
            ops += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
            views.append(transforms.Compose(ops))
    return views


def load_model(checkpoint_path, device, fuse=True):
    """Load a checkpoint saved by `train.train` and return (forward function, class names)."""
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model = VanillaNet7(num_classes=len(ckpt["class_names"]), width=ckpt["config"]["width"])
    model.load_state_dict(ckpt["ema_state_dict"])
    model.eval()
    if fuse:
        return fuse_model(model).to(device), ckpt["class_names"]
    # Deep training ends with lmbd = 1, which is also what fusion assumes
    return partial(model.to(device), lmbd=1.0), ckpt["class_names"]


@torch.no_grad()
def predict_logits(forward, images, tta_views, device, batch_size=64):
    """Sum of the logits over all TTA views, for uint8 images of shape (N, H, W)."""
    all_logits = []
    for start in range(0, len(images), batch_size):
        pil_images = [Image.fromarray(img) for img in images[start:start + batch_size]]
        batch_logits = 0
        for view in tta_views:
            batch = torch.stack([view(img) for img in pil_images]).to(device)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                batch_logits = batch_logits + forward(batch).float()
        all_logits.append(batch_logits)
    return torch.cat(all_logits)


def predict(checkpoint_paths, images, device, fuse=True, crop_ratio=0.9, batch_size=64):
    """Predicted class indices, averaging the TTA logits of every checkpoint."""
    tta_views = build_tta_transforms(crop_ratio)
    total = 0
    for path in checkpoint_paths:
        forward, _ = load_model(path, device, fuse)
        total = total + predict_logits(forward, images, tta_views, device, batch_size)
    return (total / len(checkpoint_paths)).argmax(dim=1).cpu().numpy()
