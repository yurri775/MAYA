"""Augment each morph identity into many images.

Given a folder containing base morph images (each image representing
une nouvelle identite fictive, par ex. A__B__a050.png), this script
applique des petites augmentations (rotation, translation, jitter de
luminosite/contraste, bruit leger) pour generer plusieurs images
par identite.

Exemple d'utilisation :

  python augment_identities.py \
      --in_dir morph_identities \
      --out_dir big_dataset \
      --n_aug 30

Cela genere, pour chaque image de base, 30 versions augmentees,
plus une copie de l'originale, dans le dossier big_dataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import random


def random_augment(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]

    # Rotation legere
    angle = random.uniform(-5.0, 5.0)
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    aug = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

    # Translation legere
    tx = random.uniform(-0.03 * w, 0.03 * w)
    ty = random.uniform(-0.03 * h, 0.03 * h)
    M_t = np.float32([[1, 0, tx], [0, 1, ty]])
    aug = cv2.warpAffine(aug, M_t, (w, h), borderMode=cv2.BORDER_REFLECT_101)

    # Jitter de luminosite / contraste
    alpha = random.uniform(0.9, 1.1)  # contraste
    beta = random.uniform(-12, 12)    # luminosite
    aug = cv2.convertScaleAbs(aug, alpha=alpha, beta=beta)

    # Bruit gaussien leger
    noise = np.random.normal(0, 4, aug.shape).astype(np.float32)
    aug_f = np.clip(aug.astype(np.float32) + noise, 0, 255)
    aug = aug_f.astype(np.uint8)

    # Eventuellement un leger flou (rarement)
    if random.random() < 0.2:
        aug = cv2.GaussianBlur(aug, (3, 3), 0.6)

    return aug


def main() -> None:
    parser = argparse.ArgumentParser(description="Augment morph identities")
    parser.add_argument("--in_dir", type=str, required=True,
                        help="Dossier avec les images de base (identites)")
    parser.add_argument("--out_dir", type=str, default="big_dataset",
                        help="Dossier de sortie pour toutes les images fictives")
    parser.add_argument("--n_aug", type=int, default=30,
                        help="Nombre d'augmentations par image de base")

    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exts = {".png", ".jpg", ".jpeg"}
    paths = [p for p in in_dir.iterdir() if p.suffix.lower() in exts]
    paths.sort()

    total = 0

    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            print(f"[WARN] Impossible de lire {p}")
            continue

        # Copier aussi l'image originale (version _orig)
        orig_name = f"{p.stem}_orig{p.suffix}"
        cv2.imwrite(str(out_dir / orig_name), img)
        total += 1

        # Generer n_aug versions
        for k in range(1, args.n_aug + 1):
            aug = random_augment(img)
            out_name = f"{p.stem}_aug{k:03d}{p.suffix}"
            cv2.imwrite(str(out_dir / out_name), aug)
            total += 1

    print(f"Done. Generated {total} images in {out_dir}.")


if __name__ == "__main__":
    main()
