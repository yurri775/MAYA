# -*- coding: utf-8 -*-
"""
Script pour generer des echantillons de demonstration pour le professeur
Base sur les techniques du papier SynMorph
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import cv2
import dlib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.datasets import fetch_lfw_people
import time
from datetime import datetime

# ==================== CONFIGURATION ====================
SAMPLE_DIR = Path("./sample_data")
BEFORE_DIR = SAMPLE_DIR / "before_morph"
AFTER_DIR = SAMPLE_DIR / "after_morph"
COMPARISON_DIR = SAMPLE_DIR / "morph_comparison"
GIFS_DIR = SAMPLE_DIR / "gifs_demo"

# Créer les dossiers
for dir_path in [BEFORE_DIR, AFTER_DIR, COMPARISON_DIR, GIFS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

LOCAL_DATA_DIR = Path("./dlib_models")
PREDICTOR_PATH = LOCAL_DATA_DIR / "shape_predictor_68_face_landmarks.dat"

SIZE = 256  # Taille pour les démonstrations
NUM_SAMPLES = 5  # Nombre d'échantillons à générer

print("""
╔═══════════════════════════════════════════════════════════════════════╗
║                                                                       ║
║        📊 GÉNÉRATION D'ÉCHANTILLONS DE DÉMONSTRATION 📊              ║
║                                                                       ║
║              Basé sur le papier SynMorph (arXiv:2409.05595)          ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝

⚙️  Configuration:
   • Nombre d'échantillons : {NUM_SAMPLES}
   • Taille des images     : {SIZE}x{SIZE}
   • Dossier de sortie     : {SAMPLE_DIR}

📁 Structure des dossiers:
   ├── before_morph/       (Images originales - Paire A et B)
   ├── after_morph/        (Images morphées)
   ├── morph_comparison/   (Comparaisons côte-à-côte)
   └── gifs_demo/          (Animations GIF)
""".format(NUM_SAMPLES=NUM_SAMPLES, SIZE=SIZE, SAMPLE_DIR=SAMPLE_DIR))

# ==================== CHARGEMENT DES MODÈLES ====================
print("\n📥 Chargement des modèles...")

try:
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(str(PREDICTOR_PATH))
    print("✓ Dlib chargé")
except Exception as e:
    print(f"❌ Erreur: {e}")
    print("⚠️  Assurez-vous que le modèle Dlib est téléchargé!")
    exit(1)

# Charger dataset LFW
print("📥 Chargement du dataset LFW...")
lfw_people = fetch_lfw_people(min_faces_per_person=30, resize=0.5, color=False)
images = lfw_people.images
labels = lfw_people.target
target_names = lfw_people.target_names
print(f"✓ Dataset chargé: {len(images)} images, {len(target_names)} personnes")

# ==================== FONCTIONS DE MORPHING ====================

def get_landmarks(img_gray, detector, predictor):
    """Détecte les landmarks faciaux"""
    dets = detector(img_gray, 0)
    if len(dets) == 0:
        return None
    shape = predictor(img_gray, dets[0])
    pts = np.zeros((68, 2), dtype=np.int32)
    for i in range(68):
        pts[i] = (shape.part(i).x, shape.part(i).y)
    return pts

def add_corner_points(points, w, h):
    corners = np.array([
        [0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1],
        [w // 2, 0], [w - 1, h // 2], [w // 2, h - 1], [0, h // 2]
    ], dtype=np.int32)
    return np.concatenate([points, corners], axis=0)

def clamp_points(points, w, h):
    pts = np.array(points, dtype=np.float32)
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    return pts

def find_point_index(points, pt, tol=3.0):
    pts = np.asarray(points, dtype=np.float32)
    dists = np.linalg.norm(pts - np.asarray(pt, dtype=np.float32), axis=1)
    idx = int(np.argmin(dists))
    if dists[idx] <= tol:
        return idx
    return None

def triangle_completely_inside(t, w, h):
    for (x, y) in t:
        if x < 0 or x >= w or y < 0 or y >= h:
            return False
    return True

def apply_affine_transform(src, src_tri, dst_tri, size):
    warp_mat = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))
    dst = cv2.warpAffine(src, warp_mat, (int(size[0]), int(size[1])),
                         None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return dst

def morph_triangle(img1, img2, img_morphed, t1, t2, t_morphed, alpha):
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    r = cv2.boundingRect(np.float32([t_morphed]))

    if r1[2] <= 0 or r1[3] <= 0 or r2[2] <= 0 or r2[3] <= 0 or r[2] <= 0 or r[3] <= 0:
        return

    t1_rect = [(t1[i][0] - r1[0], t1[i][1] - r1[1]) for i in range(3)]
    t2_rect = [(t2[i][0] - r2[0], t2[i][1] - r2[1]) for i in range(3)]
    t_rect = [(t_morphed[i][0] - r[0], t_morphed[i][1] - r[1]) for i in range(3)]

    img1_rect = img1[r1[1]:r1[1]+r1[3], r1[0]:r1[0]+r1[2]]
    img2_rect = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]]

    if img1_rect.size == 0 or img2_rect.size == 0:
        return

    size_rect = (r[2], r[3])

    warp_img1 = apply_affine_transform(img1_rect, t1_rect, t_rect, size_rect)
    warp_img2 = apply_affine_transform(img2_rect, t2_rect, t_rect, size_rect)

    img_rect = (1.0 - alpha) * warp_img1 + alpha * warp_img2

    mask = np.zeros((r[3], r[2]), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t_rect), 1.0, 16, 0)

    y, x, w_rect, h_rect = r[1], r[0], r[2], r[3]
    img_morphed[y:y+h_rect, x:x+w_rect] = img_morphed[y:y+h_rect, x:x+w_rect] * (1 - mask[:, :, None]) + img_rect * mask[:, :, None]

def prepare_points_for_image(img_gray, detector, predictor, w, h):
    pts = get_landmarks(img_gray, detector, predictor)
    if pts is None:
        grid_x = np.tile(np.linspace(w*0.25, w*0.75, 17), (4,))
        grid_y = np.repeat(np.linspace(h*0.25, h*0.75, 4), 17)
        grid = np.vstack([grid_x[:68], grid_y[:68]]).T.astype(np.int32)
        pts = grid
    pts = clamp_points(pts, w, h)
    pts = add_corner_points(pts.astype(np.int32), w, h)
    return pts.astype(np.float32)

def morph_faces(imgA, imgB, alpha=0.5):
    """Morphe deux visages"""
    # Redimensionner
    imgA_resized = cv2.resize(imgA, (SIZE, SIZE), interpolation=cv2.INTER_CUBIC)
    imgB_resized = cv2.resize(imgB, (SIZE, SIZE), interpolation=cv2.INTER_CUBIC)

    # Convertir en uint8
    if imgA_resized.dtype != np.uint8:
        imgA_resized = (imgA_resized * 255).astype(np.uint8)
    if imgB_resized.dtype != np.uint8:
        imgB_resized = (imgB_resized * 255).astype(np.uint8)

    # Convertir en couleur
    imgA_color = cv2.cvtColor(imgA_resized, cv2.COLOR_GRAY2BGR).astype(np.float32)
    imgB_color = cv2.cvtColor(imgB_resized, cv2.COLOR_GRAY2BGR).astype(np.float32)

    # Préparer les points
    ptsA = prepare_points_for_image(imgA_resized, detector, predictor, SIZE, SIZE)
    ptsB = prepare_points_for_image(imgB_resized, detector, predictor, SIZE, SIZE)

    # Points morphés
    points_morphed = (1.0 - alpha) * ptsA + alpha * ptsB
    points_morphed = clamp_points(points_morphed, SIZE, SIZE)

    # Triangulation de Delaunay
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

    # Morphing
    img_morphed = np.zeros_like(imgA_color, dtype=np.float32)

    for tri in tri_indices:
        i1, i2, i3 = tri
        tA = [ptsA[i1], ptsA[i2], ptsA[i3]]
        tB = [ptsB[i1], ptsB[i2], ptsB[i3]]
        tM = [points_morphed[i1], points_morphed[i2], points_morphed[i3]]

        if not (triangle_completely_inside(tA, SIZE, SIZE) and
                triangle_completely_inside(tB, SIZE, SIZE) and
                triangle_completely_inside(tM, SIZE, SIZE)):
            continue

        morph_triangle(imgA_color, imgB_color, img_morphed, tA, tB, tM, alpha)

    return np.clip(img_morphed, 0, 255).astype(np.uint8)

# ==================== CRÉATION DES COMPARAISONS ====================

def create_comparison_image(imgA, imgB, morph, person_a, person_b, alpha, idx):
    """Crée une image de comparaison côte-à-côte"""

    # Préparer les images
    imgA_rgb = cv2.cvtColor(cv2.resize((imgA * 255).astype(np.uint8) if imgA.max() <= 1 else imgA.astype(np.uint8), (SIZE, SIZE)), cv2.COLOR_GRAY2BGR)
    imgB_rgb = cv2.cvtColor(cv2.resize((imgB * 255).astype(np.uint8) if imgB.max() <= 1 else imgB.astype(np.uint8), (SIZE, SIZE)), cv2.COLOR_GRAY2BGR)

    # Créer une grande image
    height = SIZE + 120
    width = SIZE * 3 + 80
    comparison = np.ones((height, width, 3), dtype=np.uint8) * 255

    # Ajouter les images
    comparison[60:60+SIZE, 20:20+SIZE] = imgA_rgb
    comparison[60:60+SIZE, SIZE+40:SIZE+40+SIZE] = morph
    comparison[60:60+SIZE, 2*SIZE+60:2*SIZE+60+SIZE] = imgB_rgb

    # Ajouter du texte
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(comparison, f"Person A: {person_a[:12]}", (20, 35), font, 0.6, (50, 50, 50), 2)
    cv2.putText(comparison, f"Morph (alpha={alpha:.1f})", (SIZE+40, 35), font, 0.6, (200, 0, 0), 2)
    cv2.putText(comparison, f"Person B: {person_b[:12]}", (2*SIZE+60, 35), font, 0.6, (50, 50, 50), 2)

    # Flèches
    cv2.arrowedLine(comparison, (SIZE+10, height//2), (SIZE+30, height//2), (100, 100, 100), 3)
    cv2.arrowedLine(comparison, (2*SIZE+50, height//2), (2*SIZE+30, height//2), (100, 100, 100), 3)

    return comparison

# ==================== GÉNÉRATION DES ÉCHANTILLONS ====================

print(f"\n🎨 Génération de {NUM_SAMPLES} échantillons de démonstration...\n")

unique_labels = np.unique(labels)
np.random.seed(42)  # Pour la reproductibilité

for i in range(NUM_SAMPLES):
    print(f"[{i+1}/{NUM_SAMPLES}] Génération de l'échantillon {i+1}...", end=" ")

    # Sélectionner deux personnes différentes aléatoirement
    person_a_label, person_b_label = np.random.choice(unique_labels, 2, replace=False)

    # Récupérer les images
    imgs_a = images[labels == person_a_label]
    imgs_b = images[labels == person_b_label]

    imgA = imgs_a[np.random.randint(len(imgs_a))]
    imgB = imgs_b[np.random.randint(len(imgs_b))]

    person_a = target_names[person_a_label]
    person_b = target_names[person_b_label]

    try:
        # Générer le morphing
        morph = morph_faces(imgA, imgB, alpha=0.5)

        # Sauvegarder les images "avant"
        imgA_save = cv2.cvtColor(cv2.resize((imgA * 255).astype(np.uint8) if imgA.max() <= 1 else imgA.astype(np.uint8), (SIZE, SIZE)), cv2.COLOR_GRAY2BGR)
        imgB_save = cv2.cvtColor(cv2.resize((imgB * 255).astype(np.uint8) if imgB.max() <= 1 else imgB.astype(np.uint8), (SIZE, SIZE)), cv2.COLOR_GRAY2BGR)

        cv2.imwrite(str(BEFORE_DIR / f"sample_{i+1:02d}_personA_{person_a[:10]}.png"), imgA_save)
        cv2.imwrite(str(BEFORE_DIR / f"sample_{i+1:02d}_personB_{person_b[:10]}.png"), imgB_save)

        # Sauvegarder le morphing
        cv2.imwrite(str(AFTER_DIR / f"sample_{i+1:02d}_morph.png"), morph)

        # Créer et sauvegarder la comparaison
        comparison = create_comparison_image(imgA, imgB, morph, person_a, person_b, 0.5, i+1)
        cv2.imwrite(str(COMPARISON_DIR / f"sample_{i+1:02d}_comparison.png"), comparison)

        print("✓")

    except Exception as e:
        print(f"❌ Erreur: {e}")
        continue

print(f"""
╔═══════════════════════════════════════════════════════════════════════╗
║                    ✅ GÉNÉRATION TERMINÉE ✅                          ║
╠═══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║ 📁 Fichiers générés dans: {str(SAMPLE_DIR):38s} ║
║                                                                       ║
║ Structure:                                                            ║
║   ├── before_morph/       → {NUM_SAMPLES*2:2d} images originales (paires A & B)    ║
║   ├── after_morph/        → {NUM_SAMPLES:2d} images morphées                      ║
║   ├── morph_comparison/   → {NUM_SAMPLES:2d} comparaisons côte-à-côte             ║
║   └── gifs_demo/          → Animations (si disponibles)              ║
║                                                                       ║
║ 💡 Ces fichiers sont prêts à être présentés à votre professeur!      ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝
""")

print(f"\n⏰ Heure de fin: {datetime.now().strftime('%H:%M:%S')}")
