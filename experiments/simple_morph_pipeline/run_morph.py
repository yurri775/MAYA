"""Run a simple morph between two local faces in simple_morph.

Example (from PowerShell in this folder):

  cd C:\\Users\\papaa\\Desktop\\simple_morph
  python run_morph.py --alpha 0.5

This will morph images/donor_A.png and images/donor_B.png and save:
  outputs/A.png, outputs/B.png, outputs/morph_a050.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
import cv2

from morph_core import morph_pair


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple 2-face morph runner")
    parser.add_argument("--imgA", type=str, default="images/donor_A.png",
                        help="Path to first image (A); default: images/donor_A.png")
    parser.add_argument("--imgB", type=str, default="images/donor_B.png",
                        help="Path to second image (B); default: images/donor_B.png")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Morph weight for B (0=A, 1=B)")
    parser.add_argument("--out_dir", type=str, default="outputs",
                        help="Output directory")

    args = parser.parse_args()

    imgA_path = Path(args.imgA)
    imgB_path = Path(args.imgB)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    imgA = cv2.imread(str(imgA_path))
    imgB = cv2.imread(str(imgB_path))

    if imgA is None or imgB is None:
        raise SystemExit("Failed to read one of the input images")

    aligned_A, aligned_B, morph = morph_pair(imgA, imgB, alpha=args.alpha)

    cv2.imwrite(str(out_dir / "A.png"), aligned_A)
    cv2.imwrite(str(out_dir / "B.png"), aligned_B)
    cv2.imwrite(str(out_dir / f"morph_a{int(args.alpha*100):03d}.png"), morph)


if __name__ == "__main__":
    main()
