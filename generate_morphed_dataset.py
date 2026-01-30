"""
Script pour générer une nouvelle base de données de visages morphés
- 30 images morphées par nouvelle identité (combinaison de 2 personnes)
- Mélange fixé à 50% (alpha = 0.5)
- Format de nommage: A_B_N (A et B = identités, N = 1 à 30)
- Taille finale: K identités × 30 images
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import cv2
import dlib
from pathlib import Path
from sklearn.datasets import fetch_lfw_people
from itertools import combinations
from tqdm import tqdm
import time
from datetime import datetime
import json

# ==================== CONFIGURATION ====================
OUTPUT_DIR = Path("./morphed_dataset")
OUTPUT_DIR.mkdir(exist_ok=True)

DLIB_MODELS_DIR = Path("./dlib_models")
PREDICTOR_PATH = DLIB_MODELS_DIR / "shape_predictor_68_face_landmarks.dat"

IMAGE_SIZE = 128
ALPHA = 0.5  # Mélange fixé à 50%
NUM_VARIATIONS = 30  # 30 images par nouvelle identité

# Paramètres du dataset LFW
MIN_FACES_PER_PERSON = 30
RESIZE_FACTOR = 0.5

print("""
╔═══════════════════════════════════════════════════════════════════════╗
║                                                                       ║
║        📊 GÉNÉRATION DE DATASET DE VISAGES MORPHÉS 📊                ║
║                                                                       ║
║              Format: A_B_N (A, B = identités, N = 1-30)              ║
║                     Mélange: 50% (alpha = 0.5)                       ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝

⚙️  Configuration:
   • Images par identité morphée : {NUM_VARIATIONS}
   • Taille des images          : {IMAGE_SIZE}x{IMAGE_SIZE}
   • Mélange (alpha)            : {ALPHA}
   • Dossier de sortie          : {OUTPUT_DIR}

""".format(NUM_VARIATIONS=NUM_VARIATIONS, IMAGE_SIZE=IMAGE_SIZE, ALPHA=ALPHA, OUTPUT_DIR=OUTPUT_DIR))

# ==================== CHARGEMENT DES MODÈLES ====================
print("📥 Chargement des modèles...")

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
lfw_people = fetch_lfw_people(min_faces_per_person=MIN_FACES_PER_PERSON,
                               resize=RESIZE_FACTOR,
                               color=False)
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

def morph_faces(imgA, imgB, alpha=0.5, add_noise=False, noise_level=0.01):
    """Morphe deux visages avec possibilité d'ajouter du bruit pour créer des variations"""
    # Redimensionner
    imgA_resized = cv2.resize(imgA, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
    imgB_resized = cv2.resize(imgB, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)

    # Convertir en uint8
    if imgA_resized.dtype != np.uint8:
        imgA_resized = (imgA_resized * 255).astype(np.uint8)
    if imgB_resized.dtype != np.uint8:
        imgB_resized = (imgB_resized * 255).astype(np.uint8)

    # Convertir en couleur
    imgA_color = cv2.cvtColor(imgA_resized, cv2.COLOR_GRAY2BGR).astype(np.float32)
    imgB_color = cv2.cvtColor(imgB_resized, cv2.COLOR_GRAY2BGR).astype(np.float32)

    # Ajouter du bruit léger pour créer des variations
    if add_noise:
        noise_a = np.random.normal(0, noise_level * 255, imgA_color.shape)
        noise_b = np.random.normal(0, noise_level * 255, imgB_color.shape)
        imgA_color = np.clip(imgA_color + noise_a, 0, 255).astype(np.float32)
        imgB_color = np.clip(imgB_color + noise_b, 0, 255).astype(np.float32)

    # Préparer les points
    ptsA = prepare_points_for_image(imgA_resized, detector, predictor, IMAGE_SIZE, IMAGE_SIZE)
    ptsB = prepare_points_for_image(imgB_resized, detector, predictor, IMAGE_SIZE, IMAGE_SIZE)

    # Points morphés
    points_morphed = (1.0 - alpha) * ptsA + alpha * ptsB
    points_morphed = clamp_points(points_morphed, IMAGE_SIZE, IMAGE_SIZE)

    # Triangulation de Delaunay
    rect = (0, 0, IMAGE_SIZE, IMAGE_SIZE)
    subdiv = cv2.Subdiv2D(rect)

    for p in points_morphed:
        x, y = float(p[0]), float(p[1])
        if 0 <= x < IMAGE_SIZE and 0 <= y < IMAGE_SIZE:
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

        if not (triangle_completely_inside(tA, IMAGE_SIZE, IMAGE_SIZE) and
                triangle_completely_inside(tB, IMAGE_SIZE, IMAGE_SIZE) and
                triangle_completely_inside(tM, IMAGE_SIZE, IMAGE_SIZE)):
            continue

        morph_triangle(imgA_color, imgB_color, img_morphed, tA, tB, tM, alpha)

    return np.clip(img_morphed, 0, 255).astype(np.uint8)

# ==================== GÉNÉRATION DU DATASET ====================

def sanitize_name(name):
    """Nettoie le nom pour l'utiliser dans un nom de fichier"""
    return name.replace(" ", "_").replace(".", "")[:20]

# Générer toutes les paires possibles d'identités
unique_labels = np.unique(labels)
all_pairs = list(combinations(unique_labels, 2))

print(f"\n🎨 Génération du dataset de morphing...")
print(f"   • Nombre de paires d'identités : {len(all_pairs)}")
print(f"   • Total d'images à générer : {len(all_pairs) * NUM_VARIATIONS}\n")

# Statistiques
stats = {
    "total_pairs": len(all_pairs),
    "images_per_pair": NUM_VARIATIONS,
    "total_images": len(all_pairs) * NUM_VARIATIONS,
    "alpha": ALPHA,
    "image_size": IMAGE_SIZE,
    "successful": 0,
    "failed": 0,
    "start_time": datetime.now().isoformat(),
    "pairs_info": []
}

start_time = time.time()

# Générer les images morphées
with tqdm(total=len(all_pairs), desc="Génération des identités morphées") as pbar:
    for pair_idx, (label_a, label_b) in enumerate(all_pairs):
        # Récupérer les noms des personnes
        name_a = sanitize_name(target_names[label_a])
        name_b = sanitize_name(target_names[label_b])

        # Récupérer toutes les images de chaque personne
        imgs_a = images[labels == label_a]
        imgs_b = images[labels == label_b]

        pair_success = 0
        pair_failed = 0

        # Générer 30 variations
        for n in range(1, NUM_VARIATIONS + 1):
            try:
                # Sélectionner aléatoirement une image de chaque personne
                imgA = imgs_a[np.random.randint(len(imgs_a))]
                imgB = imgs_b[np.random.randint(len(imgs_b))]

                # Ajouter une légère variation pour chaque image (mais garder alpha à 0.5)
                add_noise = (n > 1)  # Pas de bruit pour la première image
                noise_level = 0.005 if add_noise else 0.0

                # Générer le morphing
                morph = morph_faces(imgA, imgB, alpha=ALPHA, add_noise=add_noise, noise_level=noise_level)

                # Nom du fichier: A_B_N
                filename = f"{name_a}_{name_b}_{n}.png"
                file_path = OUTPUT_DIR / filename

                # Sauvegarder
                cv2.imwrite(str(file_path), morph)

                pair_success += 1
                stats["successful"] += 1

            except Exception as e:
                pair_failed += 1
                stats["failed"] += 1
                # Continue silencieusement pour ne pas perturber la barre de progression

        # Enregistrer les infos de la paire
        stats["pairs_info"].append({
            "pair_index": pair_idx,
            "person_a": target_names[label_a],
            "person_b": target_names[label_b],
            "successful": pair_success,
            "failed": pair_failed
        })

        pbar.update(1)

elapsed_time = time.time() - start_time
stats["end_time"] = datetime.now().isoformat()
stats["elapsed_time_seconds"] = elapsed_time
stats["elapsed_time_minutes"] = elapsed_time / 60

# Sauvegarder les statistiques
stats_file = OUTPUT_DIR / "dataset_stats.json"
with open(stats_file, 'w', encoding='utf-8') as f:
    json.dump(stats, f, indent=2, ensure_ascii=False)

# Afficher le résumé
print(f"""
╔═══════════════════════════════════════════════════════════════════════╗
║                    ✅ GÉNÉRATION TERMINÉE ✅                          ║
╠═══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║ 📊 STATISTIQUES:                                                      ║
║                                                                       ║
║   • Paires d'identités créées : {len(all_pairs):6d}                             ║
║   • Images par paire          : {NUM_VARIATIONS:6d}                             ║
║   • Total images générées     : {stats["successful"]:6d}                             ║
║   • Échecs                    : {stats["failed"]:6d}                             ║
║   • Taux de réussite          : {stats["successful"]/(stats["successful"]+stats["failed"])*100:5.1f}%                          ║
║                                                                       ║
║ ⏱️  TEMPS:                                                            ║
║                                                                       ║
║   • Temps total               : {elapsed_time/60:5.1f} min                         ║
║   • Vitesse moyenne           : {stats["successful"]/elapsed_time:5.2f} images/sec              ║
║                                                                       ║
║ 📁 FICHIERS:                                                          ║
║                                                                       ║
║   • Dossier de sortie         : {str(OUTPUT_DIR):38s} ║
║   • Fichier de stats          : dataset_stats.json                   ║
║                                                                       ║
║ 💡 Format des noms: A_B_N                                             ║
║    A = Identité 1, B = Identité 2, N = 1-30                          ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝
""")

print(f"\n⏰ Heure de fin: {datetime.now().strftime('%H:%M:%S')}")
print(f"\n✅ Dataset prêt à être utilisé pour la recherche MIA!")
