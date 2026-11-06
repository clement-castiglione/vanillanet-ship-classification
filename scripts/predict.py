"""Predict the test set and write a Kaggle submission file.

Usage:
    python scripts/predict.py --checkpoint runs/vanillanet7/model.pt --images dataset/test_images.npy

Several checkpoints can be given to average their predictions.
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch

from vanillanet_ships.predict import predict


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", nargs="+", required=True)
    parser.add_argument("--images", default="dataset/test_images.npy", help="uint8 array of shape (N, 128, 192)")
    parser.add_argument("--output", default="submissions/submission.csv")
    parser.add_argument("--no-fuse", action="store_true", help="run the training-time model instead of the fused one")
    parser.add_argument("--crop-ratio", type=float, default=0.9, help="size of the TTA crops")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    images = np.load(args.images)
    print(f"Predicting {len(images)} images with {len(args.checkpoint)} checkpoint(s)")
    predictions = predict(
        args.checkpoint,
        images,
        torch.device(args.device),
        fuse=not args.no_fuse,
        crop_ratio=args.crop_ratio,
        batch_size=args.batch_size,
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    pd.DataFrame({"Category": predictions}).to_csv(args.output, index_label="ID")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
