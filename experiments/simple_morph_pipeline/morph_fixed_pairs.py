"""Morph fixed donor pairs from two folders using the simple_morph pipeline.

Objectif (conforme à la demande du professeur)
---------------------------------------------
Pour deux donneurs A et B (par ex. LeoMessi et CristianoRonaldo), on dispose
chacun de N images différentes (par ex. 30 images de A, 30 images de B).
On veut générer N images synthétiques en faisant un morphing *un à un* :

    (A_1, B_1) -> morph_1
    (A_2, B_2) -> morph_2
    ...
    (A_N, B_N) -> morph_N

Ce script fait exactement ça en réutilisant le coeur `morph_core.morph_pair`.

Usage (depuis le dossier simple_morph)
-------------------------------------

    python morph_fixed_pairs.py \
        --dirA images/Messi \
        --dirB images/Ronaldo \
        --out_dir morph_Messi_Ronaldo \
        --alpha 0.5

Le script :
- trie les fichiers images (par nom) dans dirA et dirB,
- prend le minimum de (len(dirA), len(dirB)) comme nombre de paires,
- applique `morph_pair(imgA_i, imgB_i, alpha)` pour chaque i,
- sauvegarde les morphs dans `out_dir` avec des noms explicites :
    Aname__Bname__a050_pairXX.png

On peut aussi passer plusieurs valeurs d'alpha si besoin.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import cv2

from morph_core import morph_pair


VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def list_images(folder: Path) -> List[Path]:
    """Return sorted list of image files in a folder."""
    if not folder.exists() or not folder.is_dir():
        raise SystemExit(f"Input folder does not exist or is not a directory: {folder}")

    files = [p for p in folder.iterdir() if p.suffix.lower() in VALID_EXTS]
    files.sort()
    if not files:
        raise SystemExit(f"No image files found in {folder}")
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="Morph fixed donor pairs from two folders (A_i with B_i)")
    parser.add_argument("--dirA", type=str, required=True, help="Folder with images of donor A (e.g. Messi)")
    parser.add_argument("--dirB", type=str, required=True, help="Folder with images of donor B (e.g. Ronaldo)")
    parser.add_argument("--out_dir", type=str, required=True, help="Output folder for synthetic morphs")
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[0.5],
        help="List of alpha values (0=A, 1=B) to use for each pair (default: 0.5)",
    )
    parser.add_argument(
        "--max_pairs",
        type=int,
        default=None,
        help="Optional maximum number of A_i/B_i pairs to morph (e.g. 50). If not set, uses all pairs.",
    )

    args = parser.parse_args()

    dirA = Path(args.dirA)
    dirB = Path(args.dirB)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    filesA = list_images(dirA)
    filesB = list_images(dirB)

    n_pairs = min(len(filesA), len(filesB))
    if args.max_pairs is not None:
        n_pairs = min(n_pairs, args.max_pairs)
    print(f"Found {len(filesA)} images in {dirA}")
    print(f"Found {len(filesB)} images in {dirB}")
    print(f"Will morph {n_pairs} pairs (A_i with B_i), alphas={args.alphas}")

    pair_count = 0
    saved = 0

    for i in range(n_pairs):
        imgA_path = filesA[i]
        imgB_path = filesB[i]
        pair_count += 1

        imgA = cv2.imread(str(imgA_path))
        imgB = cv2.imread(str(imgB_path))
        if imgA is None or imgB is None:
            print(f"[WARN] Skipping pair {imgA_path.name} + {imgB_path.name}: could not read image(s)")
            continue

        print(f"Pair {pair_count}: {imgA_path.name} + {imgB_path.name}")

        for alpha in args.alphas:
            try:
                alignedA, alignedB, morph = morph_pair(imgA, imgB, alpha=float(alpha))
            except Exception as e:
                print(
                    f"[WARN] Skipping pair {imgA_path.name}, {imgB_path.name} "
                    f"at alpha={alpha}: {e}"
                )
                continue

            stemA = imgA_path.stem
            stemB = imgB_path.stem
            alpha_tag = f"a{int(float(alpha) * 100):03d}"
            out_name = f"{stemA}__{stemB}__{alpha_tag}_pair{pair_count:02d}.png"
            out_path = out_dir / out_name

            cv2.imwrite(str(out_path), morph)
            saved += 1
            print(f"  -> saved {out_path.name}")

    print(f"Done. Processed {pair_count} pairs, saved {saved} morph images to {out_dir}")


if __name__ == "__main__":
    main()
