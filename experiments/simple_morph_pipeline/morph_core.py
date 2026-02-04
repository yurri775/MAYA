"""Minimal core morphing utilities.

This version is adapted to follow the same core morphing algorithm as
andrewdcampbell/face-movie:
- dlib 68-point landmarks (local model in models/)
- add boundary points around the image
- Delaunay triangulation over the (average) landmark set
- per-triangle affine warping with BORDER_REFLECT_101
- cross-dissolve the warped images for a given alpha

We use this for a *single* morph (one alpha) instead of a full video
sequence.
"""

from __future__ import annotations

import cv2
import dlib
import numpy as np
from pathlib import Path
from typing import Tuple

ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Landmark detection
# ---------------------------------------------------------------------------

PREDICTOR_PATH = ROOT / "models" / "shape_predictor_68_face_landmarks.dat"
if not PREDICTOR_PATH.exists():
    raise FileNotFoundError(
        f"Missing landmark model: {PREDICTOR_PATH}. "
        "It should have been copied from face_morpher/data/."
    )

dlib_detector = dlib.get_frontal_face_detector()
dlib_predictor = dlib.shape_predictor(str(PREDICTOR_PATH))


def detect_landmarks(img: np.ndarray) -> np.ndarray:
    """Detect 68-point landmarks with dlib.

    Returns an array of shape (68, 2).

    Raises a ValueError if no face is found. Higher-level callers that want
    to be resilient (e.g. for difficult faces) can catch this and fall back
    to a synthetic grid of points, similar to FACEMOMO's strategy.
    """
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rects = dlib_detector(rgb, 1)
    if not rects:
        raise ValueError("No face detected")

    shape = dlib_predictor(rgb, rects[0])
    pts = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)],
                   dtype=np.float32)
    return pts


# ---------------------------------------------------------------------------
# Geometry + warping (face-movie style)
# ---------------------------------------------------------------------------

from scipy.spatial import Delaunay


def get_boundary_points(h: int, w: int) -> np.ndarray:
    """Return 8 boundary points around the image (same as face-movie)."""
    boundary_pts = [
        (1, 1), (w - 1, 1), (1, h - 1), (w - 1, h - 1),
        ((w - 1) // 2, 1), (1, (h - 1) // 2),
        ((w - 1) // 2, h - 1), ((w - 1) // 2, (h - 1) // 2),
    ]
    return np.array(boundary_pts, dtype=np.float32)


def build_face_mask(landmarks: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Soft mask covering only the inner face region (not full background).

    We:
    - take only the first 68 landmarks (ignore synthetic boundary points)
    - build a convex hull
    - heavily blur it to get a smooth transition
    Returns a HxWx1 float32 mask in [0, 1].
    """
    h, w = size
    if landmarks.shape[0] >= 68:
        pts = landmarks[:68]
    else:
        pts = landmarks

    hull = cv2.convexHull(pts.astype(np.int32))
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)

    # Slightly shrink (erode) then blur the mask so that a bit less of the
    # outer contour is morphed and a bit more comes from B.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.GaussianBlur(mask, (25, 25), 0)

    mask_f = (mask.astype(np.float32) / 255.0)[..., None]
    return mask_f


def affine_transform(src: np.ndarray,
                     src_tri: np.ndarray,
                     dst_tri: np.ndarray,
                     size: Tuple[int, int]) -> np.ndarray:
    """Apply affine transform between two triangles."""
    M = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))
    dst = cv2.warpAffine(
        src,
        M,
        size,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return dst


def morph_triangle(im: np.ndarray,
                   im_out: np.ndarray,
                   src_tri: np.ndarray,
                   dst_tri: np.ndarray) -> None:
    """Warp one triangle from im into im_out using an affine transform.

    This is essentially the same as face-movie/face_morph.py, with extra
    guards to skip degenerate triangles (zero area / outside image), similar
    to the robust implementation used in FACEMOMO.
    """
    # Bounding boxes around triangles
    sr = cv2.boundingRect(np.float32([src_tri]))
    dr = cv2.boundingRect(np.float32([dst_tri]))

    # Skip degenerate rectangles (zero width/height)
    if sr[2] <= 0 or sr[3] <= 0 or dr[2] <= 0 or dr[3] <= 0:
        return

    # Triangle coordinates relative to bounding boxes
    cropped_src_tri = [
        (src_tri[i][0] - sr[0], src_tri[i][1] - sr[1])
        for i in range(3)
    ]
    cropped_dst_tri = [
        (dst_tri[i][0] - dr[0], dst_tri[i][1] - dr[1])
        for i in range(3)
    ]

    # Crop input image to source triangle bounding box
    cropped_im = im[sr[1] : sr[1] + sr[3], sr[0] : sr[0] + sr[2]]
    if cropped_im.size == 0:
        return

    # Destination mask for the triangle
    mask = np.zeros((dr[3], dr[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(
        mask,
        np.int32(cropped_dst_tri),
        (1.0, 1.0, 1.0),
        16,
        0,
    )

    size = (dr[2], dr[3])
    warped_im = affine_transform(cropped_im, cropped_src_tri, cropped_dst_tri, size)

    # Composite warped triangle into output image
    im_slice = im_out[dr[1] : dr[1] + dr[3], dr[0] : dr[0] + dr[2]]
    if im_slice.shape != warped_im.shape:
        # Should not happen, but guard against shape mismatch
        return
    im_out[dr[1] : dr[1] + dr[3], dr[0] : dr[0] + dr[2]] = (
        im_slice * (1 - mask) + warped_im * mask
    )


def warp_im(im: np.ndarray,
            src_landmarks: np.ndarray,
            dst_landmarks: np.ndarray,
            dst_triangulation: np.ndarray) -> np.ndarray:
    """Warp im from src_landmarks to dst_landmarks over given triangulation."""
    im_out = im.copy()
    for tri_indices in dst_triangulation:
        src_tri = src_landmarks[tri_indices]
        dst_tri = dst_landmarks[tri_indices]
        morph_triangle(im, im_out, src_tri, dst_tri)
    return im_out


def _prepare_landmarks_with_fallback(img: np.ndarray, H: int, W: int) -> np.ndarray:
    """Detect landmarks, with a robust fallback grid if detection fails.

    This mirrors the strategy used in FACEMOMO: if dlib fails to find a face,
    we create a regular grid of 68 points in the central region, then add
    boundary points later in morph_pair.
    """
    try:
        pts = detect_landmarks(img)
    except ValueError:
        # Fallback grid in the center (4x17 = 68 points)
        grid_x = np.tile(np.linspace(W * 0.25, W * 0.75, 17), (4,))
        grid_y = np.repeat(np.linspace(H * 0.25, H * 0.75, 4), 17)
        pts = np.vstack([grid_x[:68], grid_y[:68]]).T.astype(np.float32)
    return pts


def morph_pair(
    img_a: np.ndarray,
    img_b: np.ndarray,
    alpha: float = 0.5,
    output_size: Tuple[int, int] | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Morph two BGR images into a fictitious identity (face-movie style).

    Returns (aligned_A, aligned_B, morphed_frame).
    """
    if output_size is None:
        # height, width
        output_size = (600, 500)

    H, W = output_size

    # Resize both images to a common working canvas.
    img_a_resized = cv2.resize(img_a, (W, H), interpolation=cv2.INTER_AREA)
    img_b_resized = cv2.resize(img_b, (W, H), interpolation=cv2.INTER_AREA)

    # Landmarks (68 points) for each image, with robust fallback when
    # detection fails (e.g. strong pose, occlusions).
    pts_a = _prepare_landmarks_with_fallback(img_a_resized, H, W)
    pts_b = _prepare_landmarks_with_fallback(img_b_resized, H, W)

    # Append identical boundary points to both sets
    boundary = get_boundary_points(H, W)
    lm_a = np.vstack([pts_a, boundary])
    lm_b = np.vstack([pts_b, boundary])

    # For a single alpha, follow face-movie's logic: use intermediate
    # landmarks as in-between shape, and triangulate on the average
    # landmarks (independent of alpha).
    avg_landmarks = (lm_a + lm_b) / 2.0
    triangulation = Delaunay(avg_landmarks).simplices

    # Interpolated landmarks for this alpha
    weighted_landmarks = (1.0 - alpha) * lm_a + alpha * lm_b

    # Work in float for blending
    im1 = np.float32(img_a_resized)
    im2 = np.float32(img_b_resized)

    warped_im1 = warp_im(im1, lm_a, weighted_landmarks, triangulation)
    warped_im2 = warp_im(im2, lm_b, weighted_landmarks, triangulation)

    # Cross-dissolve only inside the face; keep background from img_b to
    # avoid "transparent" double backgrounds / hair.
    blended = (1.0 - alpha) * warped_im1 + alpha * warped_im2
    face_mask = build_face_mask(weighted_landmarks, (H, W))  # HxWx1
    background = im2  # use donor B as background
    composite = blended * face_mask + background * (1.0 - face_mask)

    # Convert to uint8 then apply a light unsharp mask to reduce blur and
    # make the morph look more like a "real" crisp photo.
    composite_u8 = np.uint8(composite)
    blurred = cv2.GaussianBlur(composite_u8, (0, 0), 1.0)
    sharpened = cv2.addWeighted(composite_u8, 1.4, blurred, -0.4, 0)

    return img_a_resized, img_b_resized, sharpened
