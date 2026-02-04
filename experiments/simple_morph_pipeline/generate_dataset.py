"""Generate many high-quality 2-face morphs from a folder of real faces.

This is the "dataset generator" equivalent for the simple_morph project.
It reuses the high-quality morph_pair() to create many fictitious identities
from pairs of real faces, similar to what was done in the R&D project but
with much better visual quality.

Example (from this folder):

  python generate_dataset.py \
      --src_dir "../data/real_faces/extracted" \
      --out_dir "dataset" \
      --max_pairs 50 \
      --alphas 0.3 0.5 0.7

This will:
- scan src_dir for image files (png/jpg/jpeg)
- form up to max_pairs unordered pairs (i < j)
- for each pair and each alpha, run morph_pair and save only the morphed face
  into out_dir.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple

import cv2
import numpy as np
import math

from morph_core import morph_pair, detect_landmarks


def find_images(src_dir: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    files = [p for p in src_dir.iterdir() if p.suffix.lower() in exts]
    files.sort()
    return files


def precompute_image_infos(files: List[Path],
                           resize_hw: Tuple[int, int] = (600, 500)
                           ) -> Tuple[List[Path], Dict[Path, Dict[str, Any]]]:
    """Precompute landmarks and basic geometry for each image.

    This lets us decide if a pair of images is geometrically compatible
    (similar scale / position / rotation) *before* we do the expensive morph.
    """
    H, W = resize_hw
    usable: List[Path] = []
    infos: Dict[Path, Dict[str, Any]] = {}

    for p in files:
        img = cv2.imread(str(p))
        if img is None:
            print(f"[WARN] Skipping unreadable image: {p}")
            continue

        try:
            img_resized = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        except Exception as e:
            print(f"[WARN] Failed to resize {p.name}: {e}")
            continue

        try:
            pts = detect_landmarks(img_resized)
        except Exception:
            print(f"[WARN] No landmarks found in {p.name}, dropping from pool")
            continue

        x_min, y_min = np.min(pts, axis=0)
        x_max, y_max = np.max(pts, axis=0)
        cx = float((x_min + x_max) / 2.0)
        cy = float((y_min + y_max) / 2.0)
        face_h = float(y_max - y_min)
        face_w = float(x_max - x_min)

        # Approximate in-plane rotation from eye line
        left_eye = pts[36]
        right_eye = pts[45]
        angle = math.degrees(
            math.atan2(float(right_eye[1] - left_eye[1]), float(right_eye[0] - left_eye[0]))
        )

        infos[p] = {
            "center": (cx, cy),
            "size": (face_h, face_w),
            "angle": angle,
        }
        usable.append(p)

    return usable, infos


def is_compatible(info_a: Dict[str, Any],
                  info_b: Dict[str, Any],
                  diag: float,
                  min_scale_ratio: float = 0.6,
                  max_center_dist: float = 0.18,
                  max_angle_diff_deg: float = 18.0,
                  ) -> bool:
    """Heuristic test whether two faces are compatible for a nice morph.

    - scale ratio: faces should have similar size
    - center distance: faces should be roughly aligned in the frame
    - angle diff: eye line should have similar tilt (pose).
    """
    (hA, wA) = info_a["size"]
    (hB, wB) = info_b["size"]
    scaleA = max(hA, wA)
    scaleB = max(hB, wB)
    if scaleA <= 0 or scaleB <= 0:
        return False
    scale_ratio = min(scaleA, scaleB) / max(scaleA, scaleB)
    if scale_ratio < min_scale_ratio:
        return False

    (cxA, cyA) = info_a["center"]
    (cxB, cyB) = info_b["center"]
    center_dist = math.hypot(cxA - cxB, cyA - cyB) / diag
    if center_dist > max_center_dist:
        return False

    angle_diff = abs(info_a["angle"] - info_b["angle"])
    if angle_diff > max_angle_diff_deg:
        return False

    return True



def main() -> None:
    parser = argparse.ArgumentParser(description="Generate many 2-face morphs")
    parser.add_argument(
        "--src_dir",
        type=str,
        default="../data/real_faces/extracted",
        help="Directory containing source real-face images",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="dataset",
        help="Output directory for morphed images",
    )
    parser.add_argument(
        "--max_pairs",
        type=int,
        default=100,
        help="Maximum number of unique image pairs to process",
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[0.5],
        help="One or more alpha values (0=A, 1=B) to use for each pair",
    )
    parser.add_argument(
        "--pair_mode",
        type=str,
        choices=["consecutive", "disjoint", "all"],
        default="consecutive",
        help=(
            "How to choose pairs: "
            "'consecutive' = (0,1), (1,2), ... (each image can appear in 2 pairs); "
            "'disjoint' = (0,1), (2,3), ... (each image appears in at most 1 pair); "
            "'all' = all i<j combinations until max_pairs is reached."
        ),
    )
    parser.add_argument(
        "--save_triptych",
        action="store_true",
        help=(
            "If set, also save a side-by-side A+B=C panel for each morph, "
            "with A, B and C concatenated horizontally."
        ),
    )

    args = parser.parse_args()

    src_dir = Path(args.src_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = find_images(src_dir)
    if len(files) < 2:
        raise SystemExit(f"Need at least 2 images in {src_dir}, found {len(files)}")

    # Precompute geometry/pose info for compatibility filtering
    usable_files, infos = precompute_image_infos(files)
    if len(usable_files) < 2:
        raise SystemExit(
            f"After landmark detection, only {len(usable_files)} usable images left."
        )

    files = usable_files

    print(f"Found {len(files)} usable images in {src_dir} after landmark check")
    print(
        f"Will generate up to {args.max_pairs} pairs, alphas={args.alphas}, "
        f"pair_mode={args.pair_mode}"
    )

    pair_count = 0
    saved_count = 0

    # Diagonal of the working canvas used in precompute_image_infos
    H_ref, W_ref = 600, 500
    diag = math.hypot(W_ref, H_ref)

    def process_pair(idx_a: int, idx_b: int) -> None:
        nonlocal pair_count, saved_count
        if pair_count >= args.max_pairs:
            return

        imgA_path = files[idx_a]
        imgB_path = files[idx_b]

        info_a = infos.get(imgA_path)
        info_b = infos.get(imgB_path)
        if info_a is None or info_b is None:
            return

        if not is_compatible(info_a, info_b, diag):
            # Pair geometry/pose too different, likely to look bad.
            # We just skip without counting it towards max_pairs.
            print(
                f"[SKIP] Incompatible pair (scale/pose): "
                f"{imgA_path.name} + {imgB_path.name}"
            )
            return

        imgA = cv2.imread(str(imgA_path))
        imgB = cv2.imread(str(imgB_path))
        if imgA is None or imgB is None:
            print(
                f"[WARN] Skipping pair with unreadable image(s): "
                f"{imgA_path}, {imgB_path}"
            )
            return

        pair_count += 1
        print(f"Pair {pair_count}: {imgA_path.name} + {imgB_path.name}")

        for alpha in args.alphas:
            try:
                _, _, morph = morph_pair(imgA, imgB, alpha=float(alpha))
            except Exception as e:
                print(
                    f"[WARN] Skipping pair {imgA_path.name}, {imgB_path.name} "
                    f"at alpha={alpha}: {e}"
                )
                continue

            stemA = imgA_path.stem
            stemB = imgB_path.stem
            alpha_tag = f"a{int(float(alpha) * 100):03d}"
            out_name = f"{stemA}__{stemB}__{alpha_tag}.png"
            out_path = out_dir / out_name

            cv2.imwrite(str(out_path), morph)
            saved_count += 1

            # Optionally also save a triptych panel A + B = C for reporting
            if args.save_triptych:
                try:
                    h, w = morph.shape[:2]
                    imgA_resized = cv2.resize(imgA, (w, h), interpolation=cv2.INTER_AREA)
                    imgB_resized = cv2.resize(imgB, (w, h), interpolation=cv2.INTER_AREA)
                    panel = cv2.hconcat([imgA_resized, imgB_resized, morph])
                    trip_name = f"{stemA}__{stemB}__{alpha_tag}_triptych.png"
                    trip_path = out_dir / trip_name
                    cv2.imwrite(str(trip_path), panel)
                except Exception as e:
                    print(
                        f"[WARN] Could not save triptych for {imgA_path.name}, {imgB_path.name} "
                        f"at alpha={alpha}: {e}"
                    )

    # Pairing strategies
    if args.pair_mode == "consecutive":
        # (0,1), (1,2), (2,3), ...
        for i in range(len(files) - 1):
            if pair_count >= args.max_pairs:
                break
            process_pair(i, i + 1)
    elif args.pair_mode == "disjoint":
        # (0,1), (2,3), (4,5), ... each image used in at most one pair
        for i in range(0, len(files) - 1, 2):
            if pair_count >= args.max_pairs:
                break
            process_pair(i, i + 1)
    else:  # "all" combinations
        for i in range(len(files)):
            if pair_count >= args.max_pairs:
                break
            for j in range(i + 1, len(files)):
                if pair_count >= args.max_pairs:
                    break
                process_pair(i, j)

    print(
        f"Done. Processed {pair_count} pairs, "
        f"saved {saved_count} morphed images to {out_dir}"
    )


if __name__ == "__main__":
    main()
