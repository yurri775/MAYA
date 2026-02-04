"""Download a free face recognition dataset (LFW) and export identities with many images.

This script uses sklearn.datasets.fetch_lfw_people (Labeled Faces in the Wild),
qui est une base de données de visages librement utilisable pour la
recherche en reconnaissance faciale.

Objectif par rapport aux consignes du professeur
------------------------------------------------
- Récupérer des identités avec *beaucoup* d'images (par ex. >= 30 par personne).
- Sauvegarder ces images dans simple_morph/images sous la forme :

    images/lfw_person_01/<index>.png
    images/lfw_person_02/<index>.png
    ...

- Ensuite, on pourra utiliser `morph_fixed_pairs.py` pour faire des morphings
  A_i avec B_i (par exemple entre lfw_person_01 et lfw_person_02).

Usage (depuis simple_morph)
---------------------------

    python download_lfw_identities.py \
        --min_faces_per_person 30 \
        --n_identities 2

"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_lfw_people
import cv2


def main() -> None:
    parser = argparse.ArgumentParser(description="Download LFW identities and export images to simple_morph/images")
    parser.add_argument(
        "--min_faces_per_person",
        type=int,
        default=30,
        help="Minimum number of images per identity (default: 30)",
    )
    parser.add_argument(
        "--n_identities",
        type=int,
        default=2,
        help="Number of identities to export (default: 2)",
    )

    args = parser.parse_args()

    print("Downloading LFW dataset via sklearn (this may take a moment)...")
    lfw = fetch_lfw_people(
        min_faces_per_person=args.min_faces_per_person,
        resize=1.0,
        color=True,
    )

    images = lfw.images  # shape (N, h, w, 3) floats in [0, 255] or [0,1]
    labels = lfw.target
    names = lfw.target_names

    # Ensure images are uint8 in [0,255]
    if images.dtype != np.uint8:
        # sklearn returns floats in [0, 255] or [0,1] depending on version
        max_val = images.max()
        if max_val <= 1.0:
            images_u8 = (images * 255.0).astype(np.uint8)
        else:
            images_u8 = images.astype(np.uint8)
    else:
        images_u8 = images

    unique_labels, counts = np.unique(labels, return_counts=True)
    # Sort identities by number of images (descending)
    sorted_idx = np.argsort(counts)[::-1]

    sm_root = Path(__file__).resolve().parent
    images_root = sm_root / "images"
    images_root.mkdir(exist_ok=True)

    n_exported_id = 0

    for rank, uidx in enumerate(sorted_idx):
        if n_exported_id >= args.n_identities:
            break

        person_label = unique_labels[uidx]
        person_name = names[person_label].decode("utf-8") if isinstance(names[person_label], bytes) else names[person_label]
        mask = labels == person_label
        idxs = np.where(mask)[0]

        print(f"Identity {rank+1}: {person_name} with {len(idxs)} images")

        # Create folder for this person
        person_slug = f"lfw_person_{n_exported_id+1:02d}"
        out_dir = images_root / person_slug
        out_dir.mkdir(parents=True, exist_ok=True)

        for k, img_idx in enumerate(idxs, start=1):
            img = images_u8[img_idx]
            # Ensure BGR for consistency with OpenCV pipeline
            if img.shape[2] == 3:
                # lfw.images is RGB; convert to BGR for saving if desired
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:
                img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            out_path = out_dir / f"img_{k:03d}.png"
            cv2.imwrite(str(out_path), img_bgr)

        n_exported_id += 1

    print(f"Exported {n_exported_id} identities into {images_root}/lfw_person_XX folders.")
    print("You can now run morph_fixed_pairs.py with, for example:")
    print("  python morph_fixed_pairs.py --dirA images/lfw_person_01 --dirB images/lfw_person_02 --out_dir morph_lfw_pair --alphas 0.5")


if __name__ == "__main__":
    main()
