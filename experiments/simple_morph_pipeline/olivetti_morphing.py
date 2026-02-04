"""Olivetti face morphing using Dlib landmarks and Delaunay triangulation.

This script is adapted from the professor's `olivetti_morphing.ipynb` so it can run
locally (without Colab) and be reused on other datasets.

Features
--------
- Loads the augmented Olivetti dataset via kagglehub when available, otherwise
  falls back to sklearn's built-in Olivetti faces.
- Uses a local Dlib 68-landmark model from `models/shape_predictor_68_face_landmarks.dat`.
- Performs Delaunay-based triangle morphing between two identities/classes.
- Shows and also saves the source A, morphed, and source B images.

Usage (from simple_morph folder)
--------------------------------

  python olivetti_morphing.py \
      --class_a 0 --class_b 1 --alpha 0.5

Optional arguments allow you to choose specific sample indices within each
class and the output directory.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import cv2
import dlib
import matplotlib.pyplot as plt

# ------------------ Configuration defaults ------------------

ORIG_SIZE = 64
UPSCALE = 4
SIZE = ORIG_SIZE * UPSCALE  # 256


def resolve_predictor_path() -> str:
    """Return the local path to shape_predictor_68_face_landmarks.dat.

    Expects the file under `models/` relative to this script.
    """

    here = Path(__file__).resolve().parent
    predictor = here / "models" / "shape_predictor_68_face_landmarks.dat"
    if not predictor.exists():
        raise FileNotFoundError(
            f"Cannot find Dlib predictor at {predictor}.\n"
            "Please make sure shape_predictor_68_face_landmarks.dat is present in models/."
        )
    return str(predictor)


# ------------------ Data loading ------------------


def load_olivetti() -> Tuple[np.ndarray, np.ndarray]:
    """Load (images, labels) for Olivetti faces.

    First tries the augmented Kaggle dataset via kagglehub. If that fails,
    falls back to sklearn's built-in Olivetti faces.

    Returns
    -------
    images : np.ndarray of shape (N, 64, 64), dtype uint8
    labels : np.ndarray of shape (N,), dtype int
    """

    # Try kagglehub augmented dataset
    try:
        import kagglehub  # type: ignore

        folder = kagglehub.dataset_download("martininf1n1ty/olivetti-faces-augmented-dataset")
        data_path = os.path.join(folder, "augmented_faces.npy")
        label_path = os.path.join(folder, "augmented_labels.npy")
        data_flat = np.load(data_path)
        images = (data_flat.reshape(-1, 64, 64) * 255).astype(np.uint8)
        labels = np.load(label_path).astype(int)
        print("Using Kaggle augmented Olivetti dataset via kagglehub.")
        print(f"Dataset path: {folder}")
        print(f"Images loaded: {images.shape}, labels: {labels.shape}")
        return images, labels
    except Exception as e:
        print("Warning: kagglehub augmented dataset unavailable (", e, ")")
        print("Falling back to sklearn.datasets.fetch_olivetti_faces().")

    # Fallback: sklearn built-in dataset
    try:
        from sklearn.datasets import fetch_olivetti_faces  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "scikit-learn is required for the fallback Olivetti loader. "
            "Install it with `pip install scikit-learn` and try again."
        ) from e

    dataset = fetch_olivetti_faces()
    # dataset.images is (400, 64, 64) floats in [0,1]
    images = (dataset.images * 255).astype(np.uint8)
    labels = dataset.target.astype(int)
    print("Using sklearn Olivetti faces dataset.")
    print(f"Images loaded: {images.shape}, labels: {labels.shape}")
    return images, labels


# ------------------ Dlib / landmark utilities ------------------


def get_landmarks(img_gray: np.ndarray, detector, predictor, upsample_times: int = 1):
    """Return (68,2) landmarks or None if detection fails."""

    dets = detector(img_gray, upsample_times)
    if len(dets) == 0:
        return None
    shape = predictor(img_gray, dets[0])
    pts = np.zeros((68, 2), dtype=np.int32)
    for i in range(68):
        pts[i] = (shape.part(i).x, shape.part(i).y)
    return pts


def add_corner_points(points: np.ndarray, w: int, h: int) -> np.ndarray:
    """Append 8 border points: 4 corners + 4 edge centers."""

    corners = np.array(
        [
            [0, 0],
            [w - 1, 0],
            [w - 1, h - 1],
            [0, h - 1],
            [w // 2, 0],
            [w - 1, h // 2],
            [w // 2, h - 1],
            [0, h // 2],
        ],
        dtype=np.int32,
    )
    return np.concatenate([points, corners], axis=0)


def clamp_points(points: np.ndarray, w: int, h: int) -> np.ndarray:
    pts = np.array(points, dtype=np.float32)
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    return pts


def find_point_index(points: np.ndarray, pt, tol: float = 3.0):
    """Find index of the closest point to pt, or None if distance > tol."""

    pts = np.asarray(points, dtype=np.float32)
    dists = np.linalg.norm(pts - np.asarray(pt, dtype=np.float32), axis=1)
    idx = int(np.argmin(dists))
    if dists[idx] <= tol:
        return idx
    return None


def triangle_completely_inside(t, w: int, h: int) -> bool:
    for (x, y) in t:
        if x < 0 or x >= w or y < 0 or y >= h:
            return False
    return True


# ------------------ Triangle morphing core ------------------


def apply_affine_transform(src, src_tri, dst_tri, size):
    warp_mat = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))
    dst = cv2.warpAffine(
        src,
        warp_mat,
        (int(size[0]), int(size[1])),
        None,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return dst


def morph_triangle(img1, img2, img_morphed, t1, t2, t_morphed, alpha: float):
    # bounding rects
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    r = cv2.boundingRect(np.float32([t_morphed]))

    # verify non-empty
    if r1[2] <= 0 or r1[3] <= 0 or r2[2] <= 0 or r2[3] <= 0 or r[2] <= 0 or r[3] <= 0:
        return

    # offsets
    t1_rect = [(t1[i][0] - r1[0], t1[i][1] - r1[1]) for i in range(3)]
    t2_rect = [(t2[i][0] - r2[0], t2[i][1] - r2[1]) for i in range(3)]
    t_rect = [(t_morphed[i][0] - r[0], t_morphed[i][1] - r[1]) for i in range(3)]

    # crop patches
    img1_rect = img1[r1[1] : r1[1] + r1[3], r1[0] : r1[0] + r1[2]]
    img2_rect = img2[r2[1] : r2[1] + r2[3], r2[0] : r2[0] + r2[2]]

    if img1_rect.size == 0 or img2_rect.size == 0:
        return

    size_rect = (r[2], r[3])

    warp_img1 = apply_affine_transform(img1_rect, t1_rect, t_rect, size_rect)
    warp_img2 = apply_affine_transform(img2_rect, t2_rect, t_rect, size_rect)

    img_rect = (1.0 - alpha) * warp_img1 + alpha * warp_img2

    # mask
    mask = np.zeros((r[3], r[2]), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t_rect), 1.0, 16, 0)

    # copy into output
    y, x, w_rect, h_rect = r[1], r[0], r[2], r[3]
    img_morphed[y : y + h_rect, x : x + w_rect] = (
        img_morphed[y : y + h_rect, x : x + w_rect] * (1 - mask[:, :, None])
        + img_rect * mask[:, :, None]
    )


# ------------------ High-level morph function ------------------


def prepare_points_for_image(img_gray: np.ndarray, detector, predictor, w: int, h: int) -> np.ndarray:
    """Return (76,2) points (68 landmarks + 8 boundary points) as float32.

    If Dlib fails, we use a stable grid in the center area plus corners, to
    avoid crashing.
    """

    pts = get_landmarks(img_gray, detector, predictor, upsample_times=1)
    if pts is None:
        print("Warning: no face detected by Dlib. Using placeholder grid + corners.")
        grid_x = np.tile(np.linspace(w * 0.25, w * 0.75, 17), (4,))
        grid_y = np.repeat(np.linspace(h * 0.25, h * 0.75, 4), 17)
        grid = np.vstack([grid_x[:68], grid_y[:68]]).T.astype(np.int32)
        pts = grid
    pts = clamp_points(pts, w, h)
    pts = add_corner_points(pts.astype(np.int32), w, h)
    return pts.astype(np.float32)


def morph_two_olivetti_faces(
    imgA: np.ndarray,
    imgB: np.ndarray,
    alpha: float,
    detector,
    predictor,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Morph two single-channel 64x64 Olivetti faces.

    Returns (imgA_color, imgB_color, img_morphed_color) as uint8 BGR images.
    """

    # upscale
    imgA_up = cv2.resize(imgA, (SIZE, SIZE), interpolation=cv2.INTER_CUBIC)
    imgB_up = cv2.resize(imgB, (SIZE, SIZE), interpolation=cv2.INTER_CUBIC)

    # 3-channel float
    imgA_color = cv2.cvtColor(imgA_up, cv2.COLOR_GRAY2BGR).astype(np.float32)
    imgB_color = cv2.cvtColor(imgB_up, cv2.COLOR_GRAY2BGR).astype(np.float32)

    ptsA = prepare_points_for_image(imgA_up, detector, predictor, SIZE, SIZE)
    ptsB = prepare_points_for_image(imgB_up, detector, predictor, SIZE, SIZE)

    assert ptsA.shape[0] == ptsB.shape[0], "Mismatch in number of points."

    points_morphed = (1.0 - alpha) * ptsA + alpha * ptsB
    points_morphed = clamp_points(points_morphed, SIZE, SIZE)

    # Delaunay triangulation
    rect = (0, 0, SIZE, SIZE)
    subdiv = cv2.Subdiv2D(rect)
    for p in points_morphed:
        x, y = float(p[0]), float(p[1])
        if 0 <= x < SIZE and 0 <= y < SIZE:
            subdiv.insert((x, y))

    triangle_list = subdiv.getTriangleList()

    tri_indices = []
    for t in triangle_list:
        tri_pts = [(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]
        inds = []
        valid = True
        for p in tri_pts:
            idx = find_point_index(points_morphed, p, tol=5.0)
            if idx is None:
                valid = False
                break
            inds.append(idx)
        if valid and len(set(inds)) == 3:
            tri_indices.append(tuple(inds))

    tri_indices = list(set(tri_indices))
    print(f"Valid triangles found: {len(tri_indices)}")

    img_morphed = np.zeros_like(imgA_color, dtype=np.float32)

    for tri in tri_indices:
        i1, i2, i3 = tri
        tA = [ptsA[i1], ptsA[i2], ptsA[i3]]
        tB = [ptsB[i1], ptsB[i2], ptsB[i3]]
        tM = [points_morphed[i1], points_morphed[i2], points_morphed[i3]]

        if not (
            triangle_completely_inside(tA, SIZE, SIZE)
            and triangle_completely_inside(tB, SIZE, SIZE)
            and triangle_completely_inside(tM, SIZE, SIZE)
        ):
            continue

        morph_triangle(imgA_color, imgB_color, img_morphed, tA, tB, tM, alpha)

    # Convert to uint8 for display/saving
    imgA_show = np.clip(imgA_color, 0, 255).astype(np.uint8)
    imgB_show = np.clip(imgB_color, 0, 255).astype(np.uint8)
    imgM_show = np.clip(img_morphed, 0, 255).astype(np.uint8)

    return imgA_show, imgB_show, imgM_show


# ------------------ CLI and main ------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Olivetti face morphing (professor's notebook port)")
    p.add_argument("--class_a", type=int, default=0, help="Label/class index for source A")
    p.add_argument("--class_b", type=int, default=1, help="Label/class index for source B")
    p.add_argument("--alpha", type=float, default=0.5, help="Morph ratio between 0 and 1")
    p.add_argument(
        "--index_a",
        type=int,
        default=0,
        help="Which sample index within class A to use (0-based among that class)",
    )
    p.add_argument(
        "--index_b",
        type=int,
        default=0,
        help="Which sample index within class B to use (0-based among that class)",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="olivetti_results",
        help="Directory where A/B/morph images will be saved",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    predictor_path = resolve_predictor_path()
    print(f"Using Dlib predictor: {predictor_path}")

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    images, labels = load_olivetti()

    # Select indices for the requested classes
    idxs_a = np.where(labels == args.class_a)[0]
    idxs_b = np.where(labels == args.class_b)[0]
    if len(idxs_a) == 0 or len(idxs_b) == 0:
        raise SystemExit(
            f"Class {args.class_a} or {args.class_b} not found in labels. "
            f"len(idxs_a)={len(idxs_a)}, len(idxs_b)={len(idxs_b)}"
        )

    if args.index_a >= len(idxs_a) or args.index_b >= len(idxs_b):
        raise SystemExit(
            f"index_a/index_b out of range for selected classes. "
            f"len(idxs_a)={len(idxs_a)}, len(idxs_b)={len(idxs_b)}"
        )

    idx_a = idxs_a[args.index_a]
    idx_b = idxs_b[args.index_b]

    imgA = images[idx_a]
    imgB = images[idx_b]

    print(f"Selected A: global index {idx_a}, class {args.class_a}, class-idx {args.index_a}")
    print(f"Selected B: global index {idx_b}, class {args.class_b}, class-idx {args.index_b}")

    imgA_show, imgB_show, imgM_show = morph_two_olivetti_faces(
        imgA, imgB, alpha=args.alpha, detector=detector, predictor=predictor
    )

    # Downscale for compact display (optional)
    imgA_disp = cv2.resize(imgA_show, (ORIG_SIZE, ORIG_SIZE), interpolation=cv2.INTER_AREA)
    imgB_disp = cv2.resize(imgB_show, (ORIG_SIZE, ORIG_SIZE), interpolation=cv2.INTER_AREA)
    imgM_disp = cv2.resize(imgM_show, (ORIG_SIZE, ORIG_SIZE), interpolation=cv2.INTER_AREA)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.title(f"Source A (class {args.class_a})")
    plt.imshow(cv2.cvtColor(imgA_disp, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title(f"Morphed (alpha={args.alpha:.2f})")
    plt.imshow(cv2.cvtColor(imgM_disp, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title(f"Source B (class {args.class_b})")
    plt.imshow(cv2.cvtColor(imgB_disp, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    # Save images to disk for the professor
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    alpha_int = int(round(args.alpha * 100))

    path_A = out_dir / f"olivetti_A_class{args.class_a}_idx{args.index_a}.png"
    path_B = out_dir / f"olivetti_B_class{args.class_b}_idx{args.index_b}.png"
    path_M = out_dir / f"olivetti_morph_a{args.class_a}_b{args.class_b}_alpha{alpha_int:03d}.png"

    cv2.imwrite(str(path_A), imgA_show)
    cv2.imwrite(str(path_B), imgB_show)
    cv2.imwrite(str(path_M), imgM_show)

    print(f"Saved A image to {path_A}")
    print(f"Saved B image to {path_B}")
    print(f"Saved morphed image to {path_M}")


if __name__ == "__main__":
    main()
