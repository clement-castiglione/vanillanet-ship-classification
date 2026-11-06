"""Convert the training images (one folder per class) into a single .npz archive.

Usage:
    python scripts/prepare_data.py --images-dir dataset/ships_gray/ships_gray --output data/train_set.npz
"""

import argparse

from vanillanet_ships.data import create_npz_from_images


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images-dir", required=True, help="folder containing one sub-folder of images per class")
    parser.add_argument("--output", default="data/train_set.npz")
    args = parser.parse_args()

    class_names = create_npz_from_images(args.images_dir, args.output)
    print(f"Saved {args.output} with {len(class_names)} classes: {', '.join(class_names)}")


if __name__ == "__main__":
    main()
