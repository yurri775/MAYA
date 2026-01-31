"""
Script pour générer une base de données de visages morphés
Basé sur morph1.ipynb - Adapté pour CASIA-WebFace

Configuration:
- 50 personnes sélectionnées (celles avec le plus d'images)
- 30 images morphées par paire d'identités
- 1225 paires × 30 = 36,750 images total
- Alpha fixé à 0.5 (50%)
- Format de nommage: A_B_N.png (A et B = identités, N = 1 à 30)
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import cv2
import dlib
import os
from pathlib import Path
import urllib.request
import bz2
from itertools import combinations
from tqdm import tqdm
import time
from datetime import datetime
import json
from collections import defaultdict

# ==================== CONFIGURATION ====================


# Dossier du dataset CASIA-WebFace
CASIA_DIR = Path(r"C:\Users\marwa\OneDrive\Desktop\moprh\casia_webface\data tri\images_validees")

# Dossier de sortie
OUTPUT_DIR = Path("./morphed_dataset_final")
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Modèle Dlib
LOCAL_DATA_DIR = Path("./dlib_models")
LOCAL_DATA_DIR.mkdir(exist_ok=True)
PREDICTOR_PATH = LOCAL_DATA_DIR / "shape_predictor_68_face_landmarks.dat"

# Paramètres
SIZE = 256                 # Taille finale des images
ALPHA = 0.5               # Mélange fixé à 50%
NUM_VARIATIONS = 30       # 30 images par paire d'identités
MAX_PERSONS = 50          # 50 personnes → 1225 paires
MIN_IMAGES_PER_PERSON = 10  # Minimum d'images par personne

print("="*70)
print("GENERATION DE DATASET MORPHE - CASIA-WebFace")
print("="*70)
print(f"""
Configuration:
   - Personnes selectionnees: {MAX_PERSONS}
   - Paires d'identites: {MAX_PERSONS * (MAX_PERSONS - 1) // 2}
   - Images par paire: {NUM_VARIATIONS}
   - Total images: {MAX_PERSONS * (MAX_PERSONS - 1) // 2 * NUM_VARIATIONS}
   - Alpha (melange): {ALPHA} (50%)
   - Taille images: {SIZE}x{SIZE}
   - Format: A_B_N.png
   - Dossier sortie: {OUTPUT_DIR}
""")

# ==================== TÉLÉCHARGER DLIB ====================

def download_dlib_predictor():
    """Telecharge le fichier shape_predictor_68_face_landmarks.dat si necessaire"""
    if PREDICTOR_PATH.exists():
        print("[OK] Modele Dlib deja present")
        return

    print("[...] Telechargement du modele Dlib...")
    url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    compressed_file = LOCAL_DATA_DIR / "shape_predictor_68_face_landmarks.dat.bz2"

    try:
        urllib.request.urlretrieve(url, compressed_file)
        print("   Decompression...")

        with bz2.BZ2File(compressed_file, 'rb') as f_in:
            with open(PREDICTOR_PATH, 'wb') as f_out:
                f_out.write(f_in.read())

        compressed_file.unlink()
        print("[OK] Modele Dlib pret")

    except Exception as e:
        raise RuntimeError(f"[ERREUR] {e}")

# Telecharger et charger Dlib
download_dlib_predictor()

try:
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(str(PREDICTOR_PATH))
    print("[OK] Modele Dlib charge")
except Exception as e:
    raise RuntimeError(f"[ERREUR] Impossible de charger Dlib: {e}")

# ==================== CHARGER CASIA-WebFace ====================

def load_casia_dataset():
    """Charge le dataset CASIA-WebFace - Structure plate PERSONID_IMAGEID.jpg"""
    print("\n[INFO] Chargement du dataset CASIA-WebFace...")

    persons = defaultdict(list)

    if not CASIA_DIR.exists():
        raise RuntimeError(f"[ERREUR] Dossier non trouve: {CASIA_DIR}")

    # Lister toutes les images (structure plate)
    all_images = list(CASIA_DIR.glob("*.jpg"))
    print(f"   {len(all_images)} images trouvees")

    # Grouper par person_id (format: PERSONID_IMAGEID.jpg)
    for img_path in tqdm(all_images, desc="Chargement"):
        parts = img_path.stem.split("_")
        if len(parts) >= 2:
            person_id = parts[0]
            persons[person_id].append(str(img_path))

    print(f"   {len(persons)} personnes identifiees")

    # Filtrer les personnes avec assez d'images
    valid_persons = {k: v for k, v in persons.items() if len(v) >= MIN_IMAGES_PER_PERSON}

    # Trier par nombre d'images et selectionner les top MAX_PERSONS
    sorted_persons = sorted(valid_persons.items(), key=lambda x: len(x[1]), reverse=True)
    selected_persons = dict(sorted_persons[:MAX_PERSONS])

    total_images = sum(len(imgs) for imgs in selected_persons.values())
    avg_images = total_images / len(selected_persons) if selected_persons else 0

    print(f"""
[OK] Dataset charge!
   - Personnes disponibles: {len(valid_persons)}
   - Personnes selectionnees: {len(selected_persons)}
   - Total images utilisables: {total_images}
   - Moyenne images/personne: {avg_images:.1f}
""")

    return selected_persons

# ==================== FONCTIONS DE MORPHING ====================

def get_landmarks(img_gray, detector, predictor, upsample_times=0):
    """Détecte les landmarks faciaux"""
    dets = detector(img_gray, upsample_times)
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

    try:
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
    except Exception:
        pass

def prepare_points_for_image(img_gray, w, h):
    pts = get_landmarks(img_gray, detector, predictor, upsample_times=0)
    if pts is None:
        # Créer une grille par défaut
        grid_x = np.tile(np.linspace(w*0.25, w*0.75, 17), (4,))
        grid_y = np.repeat(np.linspace(h*0.25, h*0.75, 4), 17)
        grid = np.vstack([grid_x[:68], grid_y[:68]]).T.astype(np.int32)
        pts = grid
    pts = clamp_points(pts, w, h)
    pts = add_corner_points(pts.astype(np.int32), w, h)
    return pts.astype(np.float32)

def morph_faces(img_path_a, img_path_b, alpha=0.5):
    """Morphe deux visages à partir de leurs chemins"""
    # Charger les images
    imgA = cv2.imread(str(img_path_a))
    imgB = cv2.imread(str(img_path_b))

    if imgA is None or imgB is None:
        return None

    # Redimensionner
    imgA_resized = cv2.resize(imgA, (SIZE, SIZE), interpolation=cv2.INTER_LANCZOS4)
    imgB_resized = cv2.resize(imgB, (SIZE, SIZE), interpolation=cv2.INTER_LANCZOS4)

    # Convertir en grayscale pour la détection
    imgA_gray = cv2.cvtColor(imgA_resized, cv2.COLOR_BGR2GRAY)
    imgB_gray = cv2.cvtColor(imgB_resized, cv2.COLOR_BGR2GRAY)

    # Convertir en float pour le morphing
    imgA_color = imgA_resized.astype(np.float32)
    imgB_color = imgB_resized.astype(np.float32)

    # Préparer les points
    ptsA = prepare_points_for_image(imgA_gray, SIZE, SIZE)
    ptsB = prepare_points_for_image(imgB_gray, SIZE, SIZE)

    # Points morphés
    points_morphed = (1.0 - alpha) * ptsA + alpha * ptsB
    points_morphed = clamp_points(points_morphed, SIZE, SIZE)

    # Triangulation de Delaunay
    rect = (0, 0, SIZE, SIZE)
    subdiv = cv2.Subdiv2D(rect)

    for p in points_morphed:
        x, y = float(p[0]), float(p[1])
        if 0 <= x < SIZE and 0 <= y < SIZE:
            try:
                subdiv.insert((x, y))
            except:
                pass

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

# ==================== GÉNÉRATION DU DATASET ====================

def sanitize_name(name):
    """Nettoie le nom pour un nom de fichier"""
    return "".join(c if c.isalnum() or c == '_' else '' for c in str(name))[:20]

def main():
    # Charger le dataset
    persons = load_casia_dataset()

    if len(persons) < 2:
        print("❌ Pas assez de personnes dans le dataset")
        return

    # Générer toutes les paires
    person_ids = list(persons.keys())
    all_pairs = list(combinations(person_ids, 2))

    total_morphings = len(all_pairs) * NUM_VARIATIONS

    print("="*70)
    print("🚀 DÉBUT DE LA GÉNÉRATION")
    print("="*70)
    print(f"""
   - Paires d'identités: {len(all_pairs)}
   - Variations par paire: {NUM_VARIATIONS}
   - Total images à générer: {total_morphings}
   - Alpha: {ALPHA}
""")

    # Statistiques
    stats = {
        "total_pairs": len(all_pairs),
        "variations_per_pair": NUM_VARIATIONS,
        "total_expected": total_morphings,
        "alpha": ALPHA,
        "image_size": SIZE,
        "successful": 0,
        "failed": 0,
        "start_time": datetime.now().isoformat()
    }

    start_time = time.time()

    # Générer les morphings
    with tqdm(total=len(all_pairs), desc="Génération des paires", unit="paire") as pbar:
        for pair_idx, (person_a, person_b) in enumerate(all_pairs):

            name_a = sanitize_name(person_a)
            name_b = sanitize_name(person_b)

            imgs_a = persons[person_a]
            imgs_b = persons[person_b]

            # Générer 30 variations
            for n in range(1, NUM_VARIATIONS + 1):
                try:
                    # Sélectionner aléatoirement une image de chaque personne
                    img_a = np.random.choice(imgs_a)
                    img_b = np.random.choice(imgs_b)

                    # Générer le morphing
                    morph = morph_faces(img_a, img_b, alpha=ALPHA)

                    if morph is not None:
                        # Nom du fichier: A_B_N.png
                        filename = f"{name_a}_{name_b}_{n}.png"
                        filepath = OUTPUT_DIR / filename

                        cv2.imwrite(str(filepath), morph)
                        stats["successful"] += 1
                    else:
                        stats["failed"] += 1

                except Exception as e:
                    stats["failed"] += 1

            pbar.update(1)

    # Finaliser les stats
    elapsed_time = time.time() - start_time
    stats["end_time"] = datetime.now().isoformat()
    stats["elapsed_seconds"] = elapsed_time
    stats["elapsed_minutes"] = elapsed_time / 60

    # Sauvegarder les statistiques
    stats_file = OUTPUT_DIR / "dataset_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # Afficher le résumé
    success_rate = stats["successful"] / max(1, stats["successful"] + stats["failed"]) * 100

    print(f"""
╔═══════════════════════════════════════════════════════════════════════╗
║                    GÉNÉRATION TERMINÉE                         ║
╠═══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║  RÉSULTATS:                                                        ║
║                                                                       ║
║     • Paires d'identités: {len(all_pairs):>6}                                    ║
║     • Variations par paire: {NUM_VARIATIONS:>4}                                      ║
║     • Images générées:    {stats["successful"]:>6}                                    ║
║     • Échecs:             {stats["failed"]:>6}                                    ║
║     • Taux de réussite:   {success_rate:>5.1f}%                                   ║
║                                                                       ║
║    TEMPS:                                                           ║
║                                                                       ║
║     • Durée totale:       {elapsed_time/60:>5.1f} minutes                         ║
║     • Vitesse:            {stats["successful"]/max(1,elapsed_time):>5.1f} images/seconde              ║
║                                                                       ║
║   FICHIERS:                                                         ║
║                                                                       ║
║     • Dossier: {str(OUTPUT_DIR):<40}  ║
║     • Stats:   dataset_stats.json                                    ║
║                                                                       ║
║   FORMAT: A_B_N.png                                                 ║
║     A = Identité 1, B = Identité 2, N = 1 à 30                       ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝
""")

    print(f"\n✅ Dataset prêt pour la recherche MIA!")
    print(f"📁 Chemin: {OUTPUT_DIR.absolute()}")

if __name__ == "__main__":
    main()
