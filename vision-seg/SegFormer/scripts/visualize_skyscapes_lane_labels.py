#!/usr/bin/env python3
"""Overlay selected SkyScapes-Lane classes on their source images."""

import argparse
from pathlib import Path

import cv2
import numpy as np


# OpenCV BGR colors. Both dashed-line categories are yellow; long lines green.
CLASS_COLORS = {
    1: (0, 255, 255),  # Dash line
    2: (0, 255, 0),    # Long / solid line
    3: (0, 255, 255),  # Small dash line
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True,
                        help="SS_Multi_Lane dataset root containing train/val/test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.60,
                        help="Overlay opacity (default: 0.60)")
    args = parser.parse_args()

    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be between 0 and 1")

    written = 0
    for split in ("train", "val"):
        image_dir = args.input / split / "images"
        mask_dir = args.input / split / "labels" / "grayscale"
        for mask_path in sorted(mask_dir.glob("*.png")):
            image_path = image_dir / f"{mask_path.stem}.jpg"
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if image is None or mask is None:
                print(f"SKIP (missing pair): {mask_path.stem}")
                continue
            if image.shape[:2] != mask.shape:
                raise RuntimeError(f"Size mismatch: {image_path} vs {mask_path}")

            color_layer = np.zeros_like(image)
            selected = np.zeros(mask.shape, dtype=bool)
            for class_id, color in CLASS_COLORS.items():
                class_pixels = mask == class_id
                color_layer[class_pixels] = color
                selected |= class_pixels

            # Preserve all non-selected pixels exactly; blend only requested classes.
            blended = cv2.addWeighted(image, 1.0 - args.alpha, color_layer, args.alpha, 0)
            result = image.copy()
            result[selected] = blended[selected]

            output_path = args.output / split / f"{mask_path.stem}.jpg"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(output_path), result, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise RuntimeError(f"Could not write {output_path}")
            written += 1
    print(f"Wrote {written} visualizations to {args.output}")


if __name__ == "__main__":
    main()
