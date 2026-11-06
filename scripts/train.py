"""Train VanillaNet-7 on the ship dataset.

Usage:
    python scripts/train.py --config configs/vanillanet7.yaml --data data/train_set.npz --output-dir runs/vanillanet7

Rerunning the same command resumes from `<output-dir>/last.pt` if it exists.
"""

import argparse

import torch

from vanillanet_ships.data import load_npz
from vanillanet_ships.train import TrainConfig, train


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", default="data/train_set.npz", help="archive made by scripts/prepare_data.py")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = TrainConfig.from_yaml(args.config)
    images, labels, class_names = load_npz(args.data)
    train(cfg, images, labels, class_names, args.output_dir, torch.device(args.device))


if __name__ == "__main__":
    main()
