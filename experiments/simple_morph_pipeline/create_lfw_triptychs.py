"""Create A | C | B comparison panels for all LFW morph pairs.

For each identity pair (lfw_person_01 + lfw_person_02, etc.), this script:
- reads the morph images in morph_lfw_XX_YY/
- reconstructs the corresponding donor images A (from images/lfw_person_XX)
  and B (from images/lfw_person_YY)
- builds a horizontal panel [A | C | B]
- saves the result in lfw_morph_comparison/morph_lfw_XX_YY/

Usage (from simple_morph):

    python create_lfw_triptychs.py

"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2


def build_panel(img_a, img_c, img_b) -> "cv2.Mat":
    """Resize A and B to match C, then concatenate horizontally A|C|B."""
    h, w = img_c.shape[:2]

    def resize_to(img):
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

    a_r = resize_to(img_a)
    c_r = resize_to(img_c)
    b_r = resize_to(img_b)

    panel = cv2.hconcat([a_r, c_r, b_r])
    return panel


def parse_pair_dirs(root: Path) -> Tuple[Tuple[str, str, str], ...]:
    """Return list of (pair_dir_name, personA_dir_name, personB_dir_name)."""
    return (
        ("morph_lfw_01_02", "lfw_person_01", "lfw_person_02"),
        ("morph_lfw_03_04", "lfw_person_03", "lfw_person_04"),
        ("morph_lfw_05_06", "lfw_person_05", "lfw_person_06"),
        ("morph_lfw_07_08", "lfw_person_07", "lfw_person_08"),
        ("morph_lfw_09_10", "lfw_person_09", "lfw_person_10"),
    )


def main() -> None:
    root = Path(__file__).resolve().parent
    images_root = root / "images"

    out_root = root / "lfw_morph_comparison"
    out_root.mkdir(exist_ok=True)

    total_panels = 0

    for pair_dir_name, personA_name, personB_name in parse_pair_dirs(root):
        pair_dir = root / pair_dir_name
        if not pair_dir.exists():
            print(f"[WARN] {pair_dir} does not exist, skipping.")
            continue

        dirA = images_root / personA_name
        dirB = images_root / personB_name
        if not dirA.exists() or not dirB.exists():
            print(f"[WARN] Missing donor folders {dirA} or {dirB}, skipping {pair_dir_name}.")
            continue

        out_dir = out_root / pair_dir_name
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"Processing pair {pair_dir_name}: {dirA.name} + {dirB.name}")

        for morph_path in sorted(pair_dir.glob("*.png")):
            stem = morph_path.stem  # e.g. img_001__img_001__a050_pair01
            parts = stem.split("__")
            if len(parts) < 2:
                print(f"[WARN] Cannot parse donors from {morph_path.name}, skipping.")
                continue

            stemA = parts[0]
            stemB = parts[1]

            A_path = dirA / f"{stemA}.png"
            B_path = dirB / f"{stemB}.png"

            imgA = cv2.imread(str(A_path))
            imgB = cv2.imread(str(B_path))
            imgC = cv2.imread(str(morph_path))

            if imgA is None or imgB is None or imgC is None:
                print(
                    f"[WARN] Could not read A/B/C for {morph_path.name}: "
                    f"A={A_path}, B={B_path}"
                )
                continue

            panel = build_panel(imgA, imgC, imgB)

            out_name = f"panel_{morph_path.name}"
            out_path = out_dir / out_name
            cv2.imwrite(str(out_path), panel)
            total_panels += 1

    print(f"Done. Created {total_panels} A|C|B panels under {out_root}.")


if __name__ == "__main__":
    main()
