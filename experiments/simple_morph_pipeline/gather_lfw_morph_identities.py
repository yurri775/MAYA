"""Collect all LFW-based morphs into a single folder for augmentation.

This script scans the morph_lfw_*_* folders and copies all PNG morphs into
`lfw_morph_identities_base/` with unique, prefixed names so we can then run
`augment_identities.py` once to build a large synthetic dataset.

Naming convention for copied files:
    morph_01_02__<original_name>.png
    morph_03_04__<original_name>.png
    ...

Run from simple_morph:
    python gather_lfw_morph_identities.py
"""

from __future__ import annotations

from pathlib import Path
import shutil


def main() -> None:
    root = Path(__file__).resolve().parent

    # Folders we created for 5 identity pairs
    pair_dirs = [
        "morph_lfw_01_02",
        "morph_lfw_03_04",
        "morph_lfw_05_06",
        "morph_lfw_07_08",
        "morph_lfw_09_10",
    ]

    out_dir = root / "lfw_morph_identities_base"
    out_dir.mkdir(exist_ok=True)

    total = 0

    for dname in pair_dirs:
        src_dir = root / dname
        if not src_dir.exists():
            print(f"[WARN] Source dir {src_dir} does not exist, skipping.")
            continue

        prefix = dname  # e.g. morph_lfw_01_02
        for p in sorted(src_dir.glob("*.png")):
            new_name = f"{prefix}__{p.name}"
            dst = out_dir / new_name
            shutil.copy2(p, dst)
            total += 1

    print(f"Collected {total} morph images into {out_dir}.")


if __name__ == "__main__":
    main()
