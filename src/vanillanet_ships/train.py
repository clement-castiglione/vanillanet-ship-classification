"""Training loop: deep training schedule, mixup/cutmix, EMA, mixed precision.

The model that is evaluated and saved is an exponential moving average (EMA)
of the trained weights, which is more stable than the weights at any single
step.
"""

import copy
import csv
import os
import random
from dataclasses import asdict, dataclass, fields

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader

from .augment import mix_batch, smooth_labels
from .data import ShipDataset, eval_transforms, train_transforms
from .model import VanillaNet7


@dataclass
class TrainConfig:
    width: float = 1.0
    num_epochs: int = 300
    batch_size: int = 64
    lr: float = 5e-4
    weight_decay: float = 0.1
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    label_smoothing: float = 0.12
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0
    grad_accum_steps: int = 2
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    # lmbd goes linearly from 0 to 1 over these epochs, then stays at 1
    deep_training_epochs: int = 200
    # 0 trains on the whole dataset, without validation
    val_split: float = 0.0
    val_every: int = 10
    # Early stopping, counted in validation checks (only used with a validation split)
    patience: int = 14
    seed: int = 0
    num_workers: int = 4

    @classmethod
    def from_yaml(cls, path):
        with open(path) as f:
            values = yaml.safe_load(f) or {}
        types = {f.name: f.type for f in fields(cls)}
        unknown = set(values) - set(types)
        if unknown:
            raise ValueError(f"Unknown config keys in {path}: {sorted(unknown)}")
        # PyYAML reads "5e-4" as a string, hence the explicit conversion
        return cls(**{key: types[key](value) for key, value in values.items()})


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class CUDAPrefetcher:
    """Copies the next batch to the GPU on a side stream while the current one is processed."""

    def __init__(self, loader, device):
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream(device=device)

    def __iter__(self):
        first = True
        for next_images, next_labels in self.loader:
            with torch.cuda.stream(self.stream):
                next_images = next_images.to(self.device, non_blocking=True)
                next_labels = next_labels.to(self.device, non_blocking=True)
            if not first:
                yield images, labels
            first = False
            torch.cuda.current_stream(self.device).wait_stream(self.stream)
            images, labels = next_images, next_labels
        yield images, labels

    def __len__(self):
        return len(self.loader)


@torch.no_grad()
def update_ema(ema_model, model, decay):
    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
        ema_p.mul_(decay).add_(p.detach(), alpha=1 - decay)
    # BatchNorm running statistics are copied, not averaged
    for ema_b, b in zip(ema_model.buffers(), model.buffers()):
        ema_b.copy_(b)


@torch.no_grad()
def evaluate(model, batches, lmbd, device):
    """Accuracy in % of `model` over an iterable of (images, labels) batches."""
    model.eval()
    correct = total = 0
    for images, labels in batches:
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            logits = model(images, lmbd)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total


def split_train_val(images, labels, val_split, seed):
    if val_split <= 0:
        return (images, labels), None
    indices = np.random.default_rng(seed).permutation(len(images))
    n_train = int((1 - val_split) * len(images))
    train_idx, val_idx = indices[:n_train], indices[n_train:]
    return (images[train_idx], labels[train_idx]), (images[val_idx], labels[val_idx])


def make_loader(images, labels, transform, cfg, device, shuffle):
    # On Windows each worker receives a copy of the image array (~1 GB for the full dataset)
    return DataLoader(
        ShipDataset(images, labels, transform),
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        persistent_workers=cfg.num_workers > 0,
        pin_memory=device.type == "cuda",
    )


def save_model(path, ema_model, cfg, class_names, epoch, val_acc=None):
    torch.save({
        "ema_state_dict": ema_model.state_dict(),
        "config": asdict(cfg),
        "class_names": class_names,
        "epoch": epoch,
        "val_acc": val_acc,
    }, path)


def train(cfg, images, labels, class_names, output_dir, device):
    """Train a VanillaNet7 and save its EMA weights to `output_dir/model.pt`.

    With a validation split, the saved model is the best EMA checkpoint;
    without one, it is the EMA at the last epoch. A checkpoint of the full
    training state is written to `output_dir/last.pt` every `val_every` epochs,
    and training resumes from it if it exists.
    """
    os.makedirs(output_dir, exist_ok=True)
    set_seed(cfg.seed)
    # On GPU: mixed precision and batches prefetched to the device
    on_gpu = device.type == "cuda"
    num_classes = len(class_names)

    (train_images, train_labels), val_data = split_train_val(images, labels, cfg.val_split, cfg.seed)
    train_loader = make_loader(train_images, train_labels, train_transforms(), cfg, device, shuffle=True)
    val_loader = None
    if val_data is not None:
        val_loader = make_loader(*val_data, eval_transforms(), cfg, device, shuffle=False)
    if on_gpu:
        train_loader = CUDAPrefetcher(train_loader, device)
        if val_loader is not None:
            val_loader = CUDAPrefetcher(val_loader, device)
    print(f"Training on {len(train_images)} images" + (f", validating on {len(val_data[0])}" if val_data else ""))

    model = VanillaNet7(num_classes=num_classes, width=cfg.width).to(device)
    ema_model = copy.deepcopy(model).eval()

    # As in the paper: BCE on soft targets, which handles mixup/cutmix labels directly.
    # The paper uses LAMB on ImageNet but AdamW on smaller datasets, so AdamW here.
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler(device.type, enabled=on_gpu)
    # Linear warm-up, then cosine decay (scheduler stepped once per epoch)
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=cfg.warmup_epochs),
            optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, cfg.num_epochs - cfg.warmup_epochs), eta_min=cfg.min_lr),
        ],
        milestones=[cfg.warmup_epochs],
    )

    best_val_acc, checks_without_improvement, start_epoch = 0.0, 0, 0
    last_path = os.path.join(output_dir, "last.pt")
    model_path = os.path.join(output_dir, "model.pt")
    log_path = os.path.join(output_dir, "log.csv")

    if os.path.exists(last_path):
        ckpt = torch.load(last_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        ema_model.load_state_dict(ckpt["ema_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        best_val_acc = ckpt["best_val_acc"]
        checks_without_improvement = ckpt["checks_without_improvement"]
        start_epoch = ckpt["epoch"] + 1
        # Epochs run after this checkpoint are run again: drop their log rows
        with open(log_path, newline="") as f:
            rows = list(csv.reader(f))
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerows([rows[0]] + [r for r in rows[1:] if int(r[0]) <= start_epoch])
        print(f"Resumed from {last_path} at epoch {start_epoch + 1}")
    else:
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "lr", "lmbd", "val_acc_ema"])

    for epoch in range(start_epoch, cfg.num_epochs):
        model.train()
        # Weight of the identity in the activation between the two 1x1 convolutions
        lmbd = min(epoch / cfg.deep_training_epochs, 1.0)
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for step, (batch_images, batch_labels) in enumerate(train_loader):
            if not on_gpu:
                batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
            targets = smooth_labels(batch_labels, num_classes, cfg.label_smoothing)
            mixed_images, mixed_targets = mix_batch(batch_images, targets, cfg.mixup_alpha, cfg.cutmix_alpha)

            with torch.autocast(device.type, enabled=on_gpu):
                loss = criterion(model(mixed_images, lmbd), mixed_targets) / cfg.grad_accum_steps
            scaler.scale(loss).backward()

            if (step + 1) % cfg.grad_accum_steps == 0 or step + 1 == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                update_ema(ema_model, model, cfg.ema_decay)

            running_loss += loss.item() * cfg.grad_accum_steps

        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        train_loss = running_loss / len(train_loader)
        is_check_epoch = (epoch + 1) % cfg.val_every == 0 or epoch + 1 == cfg.num_epochs

        val_acc = None
        if val_loader is not None and is_check_epoch:
            val_acc = evaluate(ema_model, val_loader, lmbd, device)
            if val_acc > best_val_acc:
                best_val_acc, checks_without_improvement = val_acc, 0
                save_model(model_path, ema_model, cfg, class_names, epoch + 1, val_acc)
            else:
                checks_without_improvement += 1

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, f"{train_loss:.5f}", f"{lr:.3e}", f"{lmbd:.3f}", "" if val_acc is None else f"{val_acc:.2f}"])
        print(
            f"Epoch {epoch + 1}/{cfg.num_epochs} | loss {train_loss:.4f} | lr {lr:.2e} | lambda {lmbd:.2f}"
            + ("" if val_acc is None else f" | val acc (EMA) {val_acc:.2f}%")
        )

        if is_check_epoch:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "ema_state_dict": ema_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "best_val_acc": best_val_acc,
                "checks_without_improvement": checks_without_improvement,
            }, last_path)

        if val_loader is not None and checks_without_improvement >= cfg.patience:
            print(f"Early stopping: no improvement over {cfg.patience} validation checks")
            break

    if val_loader is None:
        save_model(model_path, ema_model, cfg, class_names, cfg.num_epochs)
        print(f"Saved final EMA weights to {model_path}")
    else:
        print(f"Best val acc (EMA): {best_val_acc:.2f}%, weights saved to {model_path}")
    return model_path
