"""Dataset loading and image transforms.

Images are 192x128 grayscale JPEGs stored as one folder per class. They are
converted once to a compressed .npz archive, which is much faster to load than
38k individual files at every run.
"""

import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMG_HEIGHT = 128
IMG_WIDTH = 192
MEAN = (0.5,)
STD = (0.5,)


def create_npz_from_images(images_dir, dest_path):
    """Read `images_dir/<class name>/*.jpg` into a single .npz archive.

    Class indices follow the alphabetical order of the folder names, which is
    the order used by the competition. Returns the class names.
    """
    class_names = sorted(os.listdir(images_dir))
    images, labels = [], []
    for label, class_name in enumerate(class_names):
        class_dir = os.path.join(images_dir, class_name)
        for file_name in sorted(os.listdir(class_dir)):
            images.append(np.array(Image.open(os.path.join(class_dir, file_name))))
            labels.append(label)

    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
    np.savez_compressed(dest_path, images=np.array(images), labels=np.array(labels), class_names=np.array(class_names))
    return class_names


def load_npz(path):
    """Return (images, labels, class_names) from an archive made by `create_npz_from_images`."""
    with np.load(path) as data:
        return data["images"], data["labels"], data["class_names"].tolist()


class ShipDataset(Dataset):
    """In-memory dataset over uint8 arrays of shape (N, H, W)."""

    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.fromarray(self.images[idx])
        if self.transform:
            image = self.transform(image)
        return image, torch.as_tensor(self.labels[idx], dtype=torch.long)


def train_transforms():
    # The paper uses RandAugment; it gave worse results here than this simpler recipe
    return transforms.Compose([
        transforms.RandomResizedCrop((IMG_HEIGHT, IMG_WIDTH), scale=(0.45, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=12, translate=(0.07, 0.07)),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
        transforms.RandomErasing(p=0.2),
    ])


def eval_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
