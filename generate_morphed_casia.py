"""
Script pour générer une base de données de visages morphés
à partir du dataset CASIA-WebFace

Source: https://www.kaggle.com/datasets/nhatdealin/casiawebface-dataset-crop
- 100,000 images
- 10,575 identités
- ~10 images par personne
- Résolution: 250x250 pixels

Configuration:
- 30 images morphées par nouvelle identité (paire A-B)
- Mélange fixé à 50% (alpha = 0.5)
- Format de nommage: A_B_N (A et B = identités, N = 1 à 30)
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import cv2
import dlib
from pathlib import Path
from itertools import combinations
from tqdm import tqdm
import time
from datetime import datetime
import json
from collections import defaultdict

# ==================== CONFIGURATION ====================

# Chemin vers le dataset CASIA-WebFace
CASIA_DIR = Path("./casia_webface")

# Dossier de sortie pour les images morphées
OUTPUT_DIR = Path("./morphed_dataset_casia")
OUTPUT_DIR.mkdir(exist_ok=True)

# Modèle Dlib
DLIB_MODELS_DIR = Path("./dlib_models")
PREDICTOR_PATH = DLIB_MODELS_DIR / "shape_predictor_68_face_landmarks.dat"

# Paramètres de morphing
IMAGE_SIZE = 256  # Taille de sortie
ALPHA = 0.5  # Mélange fixé à 50%
NUM_VARIATIONS = 30  # 30 images par nouvelle identité

# Nombre de personnes à utiliser
MAX_PERSONS = 50  # Limiter à 50 personnes pour ~1225 paires
MIN_IMAGES_PER_PERSON = 5  # Minimum d'images par personne

print("""
╔═══════════════════════════════════════════════════════════════════════╗
║                                                                       ║
║      📊 GÉNÉRATION DE DATASET MORPHÉ - CASIA-WebFace 📊              ║
║                                                                       ║
║              Format: A_B_N (A, B = identités, N = 1-30)              ║
║                     Mélange: 50% (alpha = 0.5)                       ║
║                      Résolution: 256x256 pixels                      ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝
""")

# ==================== CHARGEMENT DES MODÈLES ====================

print("📥 Chargement des modèles Dlib...")

try:
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(str(PREDICTOR_PATH))
    print("✓ Dlib chargé avec succès")
except Exception as e:
    print(f"❌ Erreur lors du chargement de Dlib: {e}")
    print("⚠️  Assurez-vous que le modèle est dans ./dlib_models/shape_predictor_68_face_landmarks.dat")
    exit(1)

# ==================== CHARGEMENT DU DATASET CASIA ====================

def load_casia_dataset(casia_dir):
    """
    Charge le dataset CASIA-WebFace
    Structure attendue: casia_webface/[person_id]/[image_files]
    """
    casia_dir = Path(casia_dir)
    persons = defaultdict(list)

    print("📂 Recherche des images CASIA-WebFace...")

    # Chercher les sous-dossiers (chaque dossier = une personne)
    # Structure CASIA: root/[id]/[images]

    # D'abord chercher dans le dossier principal
    search_dirs = [casia_dir]

    # Chercher aussi dans les sous-dossiers courants
    for subdir in ['CASIA-WebFace', 'casia', 'data', 'images', 'crop']:
        test_dir = casia_dir / subdir
        if test_dir.exists():
            search_dirs.append(test_dir)

    for search_dir in search_dirs:
        # Parcourir les dossiers de personnes
        for person_dir in search_dir.iterdir():
            if person_dir.is_dir():
                person_id = person_dir.name

                # Collecter toutes les images de cette personne
                images = []
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                    images.extend(person_dir.glob(ext))

                if len(images) >= MIN_IMAGES_PER_PERSON:
                    persons[person_id] = [str(img) for img in images]

    # Si pas de structure en dossiers, essayer de parser les noms de fichiers
    if len(persons) < 10:
        print("⚠️  Structure en dossiers non trouvée, analyse des noms de fichiers...")

        all_images = []
        for search_dir in search_dirs:
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                all_images.extend(search_dir.glob(f"**/{ext}"))

        for img_path in all_images:
            # Le nom contient souvent l'ID: "0000001_001.jpg"
            filename = img_path.stem
            parts = filename.split('_')
            if len(parts) >= 1:
                person_id = parts[0]
                persons[person_id].append(str(img_path))

    # Filtrer les personnes avec assez d'images
    filtered = {
        pid: imgs for pid, imgs in persons.items()
        if len(imgs) >= MIN_IMAGES_PER_PERSON
    }

    print(f"\n✓ Dataset CASIA chargé:")
    print(f"   • Personnes avec {MIN_IMAGES_PER_PERSON}+ images: {len(filtered)}")
    total_imgs = sum(len(imgs) for imgs in filtered.values())
    print(f"   • Total images utilisables: {total_imgs}")

    if filtered:
        avg_imgs = total_imgs / len(filtered)
        print(f"   • Moyenne images/personne: {avg_imgs:.1f}")

    return filtered


# ==================== FONCTIONS DE MORPHING ====================

def get_landmarks(img, detector, predictor):
    """Détecte les landmarks faciaux"""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    dets = detector(gray, 1)

    if len(dets) == 0:
        return None

    largest_det = max(dets, key=lambda d: (d.right() - d.left()) * (d.bottom() - d.top()))
    shape = predictor(gray, largest_det)

    pts = np.zeros((68, 2), dtype=np.int32)
    for i in range(68):
        pts[i] = (shape.part(i).x, shape.part(i).y)
    return pts


def add_boundary_points(points, w, h):
    """Ajoute des points sur les bords"""
    boundary = np.array([
        [0, 0], [w//2, 0], [w-1, 0],
        [w-1, h//2], [w-1, h-1],
        [w//2, h-1], [0, h-1], [0, h//2]
    ], dtype=np.int32)
    return np.concatenate([points, boundary], axis=0)


def create_default_landmarks(w, h):
    """Crée des landmarks par défaut si la détection échoue"""
    x = np.linspace(w * 0.2, w * 0.8, 8)
    y = np.linspace(h * 0.15, h * 0.85, 9)

    points = []
    for yi in y:
        for xi in x[:8]:
            points.append([xi, yi])

    points = np.array(points[:68], dtype=np.int32)
    while len(points) < 68:
        points = np.vstack([points, points[0:68-len(points)]])

    return points[:68]


def apply_affine_transform(src, src_tri, dst_tri, size):
    """Applique une transformation affine"""
    warp_mat = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))
    dst = cv2.warpAffine(src, warp_mat, (size[0], size[1]),
                         flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)
    return dst


def morph_triangle(img1, img2, img_morphed, t1, t2, t_morphed, alpha):
    """Morphe un triangle entre deux images"""
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

        if len(img_rect.shape) == 3:
            mask = mask[:, :, np.newaxis]

        y1, y2 = r[1], r[1] + r[3]
        x1, x2 = r[0], r[0] + r[2]

        if y2 <= img_morphed.shape[0] and x2 <= img_morphed.shape[1]:
            img_morphed[y1:y2, x1:x2] = (
                img_morphed[y1:y2, x1:x2] * (1 - mask) + img_rect * mask
            )
    except Exception:
        pass


def morph_faces(img_path_a, img_path_b, alpha=0.5, variation_seed=None):
    """Morphe deux visages"""
    imgA = cv2.imread(str(img_path_a))
    imgB = cv2.imread(str(img_path_b))

    if imgA is None or imgB is None:
        return None

    # Redimensionner
    imgA = cv2.resize(imgA, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LANCZOS4)
    imgB = cv2.resize(imgB, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LANCZOS4)

    imgA = imgA.astype(np.float32)
    imgB = imgB.astype(np.float32)

    # Variation légère
    if variation_seed is not None:
        np.random.seed(variation_seed)
        noise_level = 0.003
        imgA = np.clip(imgA + np.random.normal(0, noise_level * 255, imgA.shape), 0, 255)
        imgB = np.clip(imgB + np.random.normal(0, noise_level * 255, imgB.shape), 0, 255)

    # Landmarks
    ptsA = get_landmarks(imgA.astype(np.uint8), detector, predictor)
    ptsB = get_landmarks(imgB.astype(np.uint8), detector, predictor)

    if ptsA is None:
        ptsA = create_default_landmarks(IMAGE_SIZE, IMAGE_SIZE)
    if ptsB is None:
        ptsB = create_default_landmarks(IMAGE_SIZE, IMAGE_SIZE)

    ptsA = add_boundary_points(ptsA, IMAGE_SIZE, IMAGE_SIZE)
    ptsB = add_boundary_points(ptsB, IMAGE_SIZE, IMAGE_SIZE)

    # Points morphés
    points_morphed = ((1.0 - alpha) * ptsA + alpha * ptsB).astype(np.float32)
    points_morphed[:, 0] = np.clip(points_morphed[:, 0], 0, IMAGE_SIZE - 1)
    points_morphed[:, 1] = np.clip(points_morphed[:, 1], 0, IMAGE_SIZE - 1)

    # Triangulation
    rect = (0, 0, IMAGE_SIZE, IMAGE_SIZE)
    subdiv = cv2.Subdiv2D(rect)

    for p in points_morphed:
        try:
            subdiv.insert((float(p[0]), float(p[1])))
        except:
            pass

    triangle_list = subdiv.getTriangleList()

    def find_index(points, pt, tol=5.0):
        dists = np.linalg.norm(points - pt, axis=1)
        idx = np.argmin(dists)
        return int(idx) if dists[idx] <= tol else None

    triangles = []
    for t in triangle_list:
        pts = [(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]
        inside = all(0 <= p[0] < IMAGE_SIZE and 0 <= p[1] < IMAGE_SIZE for p in pts)
        if not inside:
            continue

        indices = []
        valid = True
        for pt in pts:
            idx = find_index(points_morphed, np.array(pt))
            if idx is None:
                valid = False
                break
            indices.append(idx)

        if valid and len(set(indices)) == 3:
            triangles.append(tuple(indices))

    triangles = list(set(triangles))

    # Morphing
    img_morphed = np.zeros_like(imgA)

    for tri in triangles:
        i1, i2, i3 = tri
        t1 = [ptsA[i1].tolist(), ptsA[i2].tolist(), ptsA[i3].tolist()]
        t2 = [ptsB[i1].tolist(), ptsB[i2].tolist(), ptsB[i3].tolist()]
        tm = [points_morphed[i1].tolist(), points_morphed[i2].tolist(), points_morphed[i3].tolist()]
        morph_triangle(imgA, imgB, img_morphed, t1, t2, tm, alpha)

    return np.clip(img_morphed, 0, 255).astype(np.uint8)


# ==================== GÉNÉRATION ====================

def sanitize_name(name):
    """Nettoie le nom pour un nom de fichier"""
    clean = "".join(c if c.isalnum() or c == '_' else '_' for c in str(name))
    return clean[:20]


def main():
    print("\n" + "="*70)
    print("📥 CHARGEMENT DU DATASET CASIA-WebFace")
    print("="*70 + "\n")

    if not CASIA_DIR.exists():
        print(f"""
❌ Le dossier CASIA n'existe pas: {CASIA_DIR}

📋 INSTRUCTIONS:
   1. Téléchargez le dataset depuis Kaggle:
      kaggle datasets download -d nhatdealin/casiawebface-dataset-crop

   2. Extrayez dans: {CASIA_DIR.absolute()}

   3. Relancez ce script
""")
        return

    # Charger le dataset
    persons = load_casia_dataset(CASIA_DIR)

    if persons is None or len(persons) < 2:
        print("❌ Pas assez de personnes dans le dataset")
        return

    # Sélectionner les personnes avec le plus d'images
    sorted_persons = sorted(persons.items(), key=lambda x: len(x[1]), reverse=True)
    selected = dict(sorted_persons[:MAX_PERSONS])

    print(f"\n✓ Sélection de {len(selected)} personnes avec le plus d'images")

    # Générer les paires
    person_ids = list(selected.keys())
    all_pairs = list(combinations(person_ids, 2))

    print(f"\n" + "="*70)
    print("🎨 GÉNÉRATION DES IMAGES MORPHÉES")
    print("="*70)
    print(f"""
⚙️  Configuration:
   • Paires d'identités: {len(all_pairs)}
   • Images par paire: {NUM_VARIATIONS}
   • Total à générer: {len(all_pairs) * NUM_VARIATIONS}
   • Taille: {IMAGE_SIZE}x{IMAGE_SIZE}
   • Alpha: {ALPHA} (50%)
   • Sortie: {OUTPUT_DIR}
""")

    # Stats
    stats = {
        "total_pairs": len(all_pairs),
        "images_per_pair": NUM_VARIATIONS,
        "total_expected": len(all_pairs) * NUM_VARIATIONS,
        "alpha": ALPHA,
        "image_size": IMAGE_SIZE,
        "successful": 0,
        "failed": 0,
        "start_time": datetime.now().isoformat(),
    }

    start_time = time.time()

    with tqdm(total=len(all_pairs), desc="Génération des morphings", unit="paire") as pbar:
        for pair_idx, (person_a, person_b) in enumerate(all_pairs):
            name_a = sanitize_name(person_a)
            name_b = sanitize_name(person_b)

            imgs_a = selected[person_a]
            imgs_b = selected[person_b]

            for n in range(1, NUM_VARIATIONS + 1):
                try:
                    img_a = np.random.choice(imgs_a)
                    img_b = np.random.choice(imgs_b)

                    variation_seed = (pair_idx * 1000 + n) if n > 1 else None

                    morph = morph_faces(img_a, img_b, alpha=ALPHA, variation_seed=variation_seed)

                    if morph is not None:
                        filename = f"{name_a}_{name_b}_{n}.png"
                        filepath = OUTPUT_DIR / filename
                        cv2.imwrite(str(filepath), morph)
                        stats["successful"] += 1
                    else:
                        stats["failed"] += 1
                except Exception:
                    stats["failed"] += 1

            pbar.update(1)

    elapsed = time.time() - start_time
    stats["elapsed_seconds"] = elapsed
    stats["end_time"] = datetime.now().isoformat()

    # Sauvegarder stats
    with open(OUTPUT_DIR / "stats.json", 'w') as f:
        json.dump(stats, f, indent=2)

    success_rate = stats["successful"] / max(1, stats["successful"] + stats["failed"]) * 100

    print(f"""
╔═══════════════════════════════════════════════════════════════════════╗
║                    ✅ GÉNÉRATION TERMINÉE ✅                          ║
╠═══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║  📊 RÉSULTATS:                                                        ║
║     • Images générées: {stats["successful"]:>6}                                    ║
║     • Échecs:          {stats["failed"]:>6}                                    ║
║     • Taux réussite:   {success_rate:>5.1f}%                                   ║
║                                                                       ║
║  ⏱️  Durée: {elapsed/60:>5.1f} min ({stats["successful"]/max(1,elapsed):>.1f} img/sec)                      ║
║                                                                       ║
║  📁 Sortie: {str(OUTPUT_DIR):<45} ║
║                                                                       ║
║  💡 Format: A_B_N.png (A=id1, B=id2, N=1-30)                         ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    main()
