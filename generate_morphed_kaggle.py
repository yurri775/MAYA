"""
Script pour générer une base de données de visages morphés
à partir du dataset Kaggle "Male Faces Image Dataset"

Source: https://www.kaggle.com/datasets/trainingdatapro/male-selfie-image-dataset
- 20 images par personne
- Haute résolution (jusqu'à 3264 x 2448)
- 110,000+ photos

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
import pandas as pd
import os
from collections import defaultdict

# ==================== CONFIGURATION ====================

# Chemin vers le dataset Kaggle téléchargé et extrait
KAGGLE_DATASET_DIR = Path("./kaggle_male_faces")  # Dossier où extraire le ZIP Kaggle

# Dossier de sortie pour les images morphées
OUTPUT_DIR = Path("./morphed_dataset_hd")
OUTPUT_DIR.mkdir(exist_ok=True)

# Modèle Dlib
DLIB_MODELS_DIR = Path("./dlib_models")
PREDICTOR_PATH = DLIB_MODELS_DIR / "shape_predictor_68_face_landmarks.dat"

# Paramètres de morphing
IMAGE_SIZE = 512  # Taille de sortie plus grande pour HD
ALPHA = 0.5  # Mélange fixé à 50%
NUM_VARIATIONS = 30  # 30 images par nouvelle identité

# Nombre de personnes à utiliser (pour limiter si nécessaire)
MAX_PERSONS = 50  # Utiliser 50 personnes max pour créer les paires
MIN_IMAGES_PER_PERSON = 10  # Minimum d'images par personne

print("""
╔═══════════════════════════════════════════════════════════════════════╗
║                                                                       ║
║     📊 GÉNÉRATION DE DATASET MORPHÉ - KAGGLE MALE FACES 📊           ║
║                                                                       ║
║              Format: A_B_N (A, B = identités, N = 1-30)              ║
║                     Mélange: 50% (alpha = 0.5)                       ║
║                  Haute Résolution: 512x512 pixels                    ║
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

# ==================== CHARGEMENT DU DATASET KAGGLE ====================

def load_kaggle_dataset(dataset_dir):
    """
    Charge le dataset Kaggle Male Faces
    Retourne un dictionnaire {person_id: [liste des chemins d'images]}
    """
    dataset_dir = Path(dataset_dir)

    # Chercher le dossier contenant les images
    possible_dirs = [
        dataset_dir / "files",
        dataset_dir / "images",
        dataset_dir,
    ]

    images_dir = None
    for d in possible_dirs:
        if d.exists() and any(d.glob("*.jpg")) or any(d.glob("*.png")):
            images_dir = d
            break

    if images_dir is None:
        # Chercher récursivement
        for d in dataset_dir.rglob("*"):
            if d.is_dir():
                jpgs = list(d.glob("*.jpg"))
                pngs = list(d.glob("*.png"))
                if len(jpgs) + len(pngs) > 0:
                    images_dir = d
                    break

    if images_dir is None:
        print(f"❌ Impossible de trouver les images dans {dataset_dir}")
        print("   Structure attendue: kaggle_male_faces/files/*.jpg")
        return None

    print(f"📁 Dossier d'images trouvé: {images_dir}")

    # Charger les métadonnées si disponibles
    metadata_file = None
    for pattern in ["metadata.csv", "*.csv"]:
        files = list(dataset_dir.glob(pattern))
        if files:
            metadata_file = files[0]
            break

    persons = defaultdict(list)

    if metadata_file and metadata_file.exists():
        print(f"📋 Chargement des métadonnées: {metadata_file}")
        try:
            df = pd.read_csv(metadata_file)

            # Identifier la colonne d'ID
            id_col = None
            for col in ['id', 'person_id', 'ID', 'subject_id', 'identity']:
                if col in df.columns:
                    id_col = col
                    break

            if id_col:
                # Grouper par ID
                for idx, row in df.iterrows():
                    person_id = str(row[id_col])

                    # Trouver le fichier image correspondant
                    # Le nom peut être dans une colonne ou basé sur l'index
                    img_name = None
                    for col in ['filename', 'file', 'image', 'photo']:
                        if col in df.columns:
                            img_name = str(row[col])
                            break

                    if img_name is None:
                        # Essayer de construire le nom
                        img_name = f"{idx}.jpg"

                    img_path = images_dir / img_name
                    if not img_path.exists():
                        # Essayer d'autres extensions
                        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                            test_path = images_dir / f"{Path(img_name).stem}{ext}"
                            if test_path.exists():
                                img_path = test_path
                                break

                    if img_path.exists():
                        persons[person_id].append(str(img_path))
            else:
                print("⚠️  Colonne d'ID non trouvée dans les métadonnées")

        except Exception as e:
            print(f"⚠️  Erreur lecture métadonnées: {e}")

    # Si pas de métadonnées ou pas assez de données, organiser par nom de fichier
    if len(persons) < 10:
        print("📂 Organisation des images par nom de fichier...")

        # Les images Kaggle sont souvent nommées: {person_id}_{photo_num}.jpg
        # ou organisées en sous-dossiers par personne

        # D'abord, vérifier les sous-dossiers
        subdirs = [d for d in images_dir.iterdir() if d.is_dir()]

        if subdirs:
            print(f"   Trouvé {len(subdirs)} sous-dossiers")
            for subdir in subdirs:
                person_id = subdir.name
                imgs = list(subdir.glob("*.jpg")) + list(subdir.glob("*.png"))
                imgs += list(subdir.glob("*.JPG")) + list(subdir.glob("*.JPEG"))
                for img in imgs:
                    persons[person_id].append(str(img))
        else:
            # Organiser par préfixe de nom de fichier
            all_images = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
            all_images += list(images_dir.glob("*.JPG")) + list(images_dir.glob("*.JPEG"))

            print(f"   Trouvé {len(all_images)} images")

            for img_path in all_images:
                # Extraire l'ID de personne du nom de fichier
                # Format typique: "12345_01.jpg" ou "person_12345_01.jpg"
                filename = img_path.stem
                parts = filename.split('_')

                if len(parts) >= 2:
                    # L'ID est probablement le premier élément numérique
                    person_id = parts[0]
                else:
                    # Utiliser les N premiers caractères
                    person_id = filename[:10]

                persons[person_id].append(str(img_path))

    # Filtrer les personnes avec assez d'images
    filtered_persons = {
        pid: imgs for pid, imgs in persons.items()
        if len(imgs) >= MIN_IMAGES_PER_PERSON
    }

    print(f"\n✓ Dataset chargé:")
    print(f"   • Personnes avec {MIN_IMAGES_PER_PERSON}+ images: {len(filtered_persons)}")
    print(f"   • Total images utilisables: {sum(len(imgs) for imgs in filtered_persons.values())}")

    return filtered_persons


# ==================== FONCTIONS DE MORPHING ====================

def get_landmarks(img, detector, predictor):
    """Détecte les landmarks faciaux sur une image couleur ou grayscale"""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    dets = detector(gray, 1)  # Upsampling pour mieux détecter

    if len(dets) == 0:
        return None

    # Prendre le plus grand visage détecté
    largest_det = max(dets, key=lambda d: (d.right() - d.left()) * (d.bottom() - d.top()))

    shape = predictor(gray, largest_det)
    pts = np.zeros((68, 2), dtype=np.int32)
    for i in range(68):
        pts[i] = (shape.part(i).x, shape.part(i).y)
    return pts


def add_boundary_points(points, w, h):
    """Ajoute des points sur les bords pour une meilleure triangulation"""
    boundary = np.array([
        [0, 0], [w//2, 0], [w-1, 0],
        [w-1, h//2], [w-1, h-1],
        [w//2, h-1], [0, h-1], [0, h//2]
    ], dtype=np.int32)
    return np.concatenate([points, boundary], axis=0)


def create_default_landmarks(w, h):
    """Crée des landmarks par défaut si la détection échoue"""
    # Grille de points couvrant le visage
    x = np.linspace(w * 0.2, w * 0.8, 8)
    y = np.linspace(h * 0.15, h * 0.85, 9)

    points = []
    for yi in y:
        for xi in x[:min(8, len(x))]:
            points.append([xi, yi])

    # Prendre 68 points
    points = np.array(points[:68], dtype=np.int32)

    # Si pas assez de points, dupliquer
    while len(points) < 68:
        points = np.vstack([points, points[0:68-len(points)]])

    return points[:68]


def apply_affine_transform(src, src_tri, dst_tri, size):
    """Applique une transformation affine à un triangle"""
    warp_mat = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))
    dst = cv2.warpAffine(src, warp_mat, (size[0], size[1]),
                         flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)
    return dst


def morph_triangle(img1, img2, img_morphed, t1, t2, t_morphed, alpha):
    """Morphe un triangle entre deux images"""
    # Bounding boxes
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    r = cv2.boundingRect(np.float32([t_morphed]))

    # Vérifier les dimensions
    if r1[2] <= 0 or r1[3] <= 0 or r2[2] <= 0 or r2[3] <= 0 or r[2] <= 0 or r[3] <= 0:
        return

    # Triangles relatifs aux bounding boxes
    t1_rect = [(t1[i][0] - r1[0], t1[i][1] - r1[1]) for i in range(3)]
    t2_rect = [(t2[i][0] - r2[0], t2[i][1] - r2[1]) for i in range(3)]
    t_rect = [(t_morphed[i][0] - r[0], t_morphed[i][1] - r[1]) for i in range(3)]

    # Extraire les régions
    try:
        img1_rect = img1[r1[1]:r1[1]+r1[3], r1[0]:r1[0]+r1[2]]
        img2_rect = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]]

        if img1_rect.size == 0 or img2_rect.size == 0:
            return

        # Transformation affine
        size_rect = (r[2], r[3])
        warp_img1 = apply_affine_transform(img1_rect, t1_rect, t_rect, size_rect)
        warp_img2 = apply_affine_transform(img2_rect, t2_rect, t_rect, size_rect)

        # Blending
        img_rect = (1.0 - alpha) * warp_img1 + alpha * warp_img2

        # Masque triangulaire
        mask = np.zeros((r[3], r[2]), dtype=np.float32)
        cv2.fillConvexPoly(mask, np.int32(t_rect), 1.0, 16, 0)

        if len(img_rect.shape) == 3:
            mask = mask[:, :, np.newaxis]

        # Appliquer le morphing
        y1, y2 = r[1], r[1] + r[3]
        x1, x2 = r[0], r[0] + r[2]

        if y2 <= img_morphed.shape[0] and x2 <= img_morphed.shape[1]:
            img_morphed[y1:y2, x1:x2] = (
                img_morphed[y1:y2, x1:x2] * (1 - mask) + img_rect * mask
            )
    except Exception:
        pass  # Ignorer les erreurs de triangles problématiques


def morph_faces(img_path_a, img_path_b, alpha=0.5, variation_seed=None):
    """
    Morphe deux visages à partir de leurs chemins d'image

    Args:
        img_path_a: Chemin vers l'image de la personne A
        img_path_b: Chemin vers l'image de la personne B
        alpha: Coefficient de mélange (0.5 = 50%)
        variation_seed: Seed pour la variation aléatoire

    Returns:
        Image morphée (numpy array) ou None si échec
    """
    # Charger les images
    imgA = cv2.imread(str(img_path_a))
    imgB = cv2.imread(str(img_path_b))

    if imgA is None or imgB is None:
        return None

    # Redimensionner en gardant les proportions
    imgA = cv2.resize(imgA, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LANCZOS4)
    imgB = cv2.resize(imgB, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LANCZOS4)

    # Convertir en float
    imgA = imgA.astype(np.float32)
    imgB = imgB.astype(np.float32)

    # Ajouter une légère variation si demandé
    if variation_seed is not None:
        np.random.seed(variation_seed)
        noise_level = 0.003  # Très léger
        imgA = np.clip(imgA + np.random.normal(0, noise_level * 255, imgA.shape), 0, 255)
        imgB = np.clip(imgB + np.random.normal(0, noise_level * 255, imgB.shape), 0, 255)

    # Détecter les landmarks
    ptsA = get_landmarks(imgA.astype(np.uint8), detector, predictor)
    ptsB = get_landmarks(imgB.astype(np.uint8), detector, predictor)

    # Utiliser des landmarks par défaut si détection échoue
    if ptsA is None:
        ptsA = create_default_landmarks(IMAGE_SIZE, IMAGE_SIZE)
    if ptsB is None:
        ptsB = create_default_landmarks(IMAGE_SIZE, IMAGE_SIZE)

    # Ajouter les points de bordure
    ptsA = add_boundary_points(ptsA, IMAGE_SIZE, IMAGE_SIZE)
    ptsB = add_boundary_points(ptsB, IMAGE_SIZE, IMAGE_SIZE)

    # Points morphés
    points_morphed = ((1.0 - alpha) * ptsA + alpha * ptsB).astype(np.float32)

    # Clamp les points
    points_morphed[:, 0] = np.clip(points_morphed[:, 0], 0, IMAGE_SIZE - 1)
    points_morphed[:, 1] = np.clip(points_morphed[:, 1], 0, IMAGE_SIZE - 1)

    # Triangulation de Delaunay sur les points morphés
    rect = (0, 0, IMAGE_SIZE, IMAGE_SIZE)
    subdiv = cv2.Subdiv2D(rect)

    for p in points_morphed:
        try:
            subdiv.insert((float(p[0]), float(p[1])))
        except:
            pass

    triangle_list = subdiv.getTriangleList()

    # Trouver les indices des triangles
    def find_index(points, pt, tol=5.0):
        dists = np.linalg.norm(points - pt, axis=1)
        idx = np.argmin(dists)
        if dists[idx] <= tol:
            return int(idx)
        return None

    triangles = []
    for t in triangle_list:
        pts = [(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]

        # Vérifier que le triangle est dans l'image
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

    # Image morphée
    img_morphed = np.zeros_like(imgA)

    # Morphing par triangle
    for tri in triangles:
        i1, i2, i3 = tri

        t1 = [ptsA[i1].tolist(), ptsA[i2].tolist(), ptsA[i3].tolist()]
        t2 = [ptsB[i1].tolist(), ptsB[i2].tolist(), ptsB[i3].tolist()]
        tm = [points_morphed[i1].tolist(), points_morphed[i2].tolist(), points_morphed[i3].tolist()]

        morph_triangle(imgA, imgB, img_morphed, t1, t2, tm, alpha)

    return np.clip(img_morphed, 0, 255).astype(np.uint8)


# ==================== GÉNÉRATION DU DATASET ====================

def sanitize_name(name):
    """Nettoie le nom pour l'utiliser dans un nom de fichier"""
    # Garder seulement les caractères alphanumériques et underscore
    clean = "".join(c if c.isalnum() or c == '_' else '_' for c in str(name))
    return clean[:30]


def main():
    """Fonction principale de génération du dataset"""

    print("\n" + "="*70)
    print("📥 CHARGEMENT DU DATASET KAGGLE")
    print("="*70 + "\n")

    # Vérifier que le dataset existe
    if not KAGGLE_DATASET_DIR.exists():
        print(f"""
❌ Le dossier du dataset n'existe pas: {KAGGLE_DATASET_DIR}

📋 INSTRUCTIONS:
   1. Téléchargez le dataset depuis:
      https://www.kaggle.com/datasets/trainingdatapro/male-selfie-image-dataset

   2. Extrayez le fichier ZIP dans le dossier:
      {KAGGLE_DATASET_DIR.absolute()}

   3. Relancez ce script
""")
        return

    # Charger le dataset
    persons = load_kaggle_dataset(KAGGLE_DATASET_DIR)

    if persons is None or len(persons) < 2:
        print("❌ Pas assez de personnes dans le dataset")
        return

    # Limiter le nombre de personnes si nécessaire
    person_ids = list(persons.keys())[:MAX_PERSONS]

    print(f"\n✓ Utilisation de {len(person_ids)} personnes pour la génération")

    # Générer toutes les paires possibles
    all_pairs = list(combinations(person_ids, 2))

    print(f"\n" + "="*70)
    print("🎨 GÉNÉRATION DES IMAGES MORPHÉES")
    print("="*70)
    print(f"""
⚙️  Configuration:
   • Nombre de paires d'identités: {len(all_pairs)}
   • Images par paire: {NUM_VARIATIONS}
   • Total images à générer: {len(all_pairs) * NUM_VARIATIONS}
   • Taille des images: {IMAGE_SIZE}x{IMAGE_SIZE}
   • Alpha (mélange): {ALPHA} (50%)
   • Dossier de sortie: {OUTPUT_DIR}
""")

    # Statistiques
    stats = {
        "total_pairs": len(all_pairs),
        "images_per_pair": NUM_VARIATIONS,
        "total_expected": len(all_pairs) * NUM_VARIATIONS,
        "alpha": ALPHA,
        "image_size": IMAGE_SIZE,
        "successful": 0,
        "failed": 0,
        "start_time": datetime.now().isoformat(),
        "pairs_info": []
    }

    start_time = time.time()

    # Barre de progression
    with tqdm(total=len(all_pairs), desc="Génération des identités morphées", unit="paire") as pbar:

        for pair_idx, (person_a, person_b) in enumerate(all_pairs):

            name_a = sanitize_name(person_a)
            name_b = sanitize_name(person_b)

            # Images disponibles pour chaque personne
            imgs_a = persons[person_a]
            imgs_b = persons[person_b]

            pair_success = 0
            pair_failed = 0

            # Générer 30 variations
            for n in range(1, NUM_VARIATIONS + 1):
                try:
                    # Sélectionner des images aléatoires pour chaque personne
                    img_a = np.random.choice(imgs_a)
                    img_b = np.random.choice(imgs_b)

                    # Seed de variation (sauf pour la première image)
                    variation_seed = (pair_idx * 1000 + n) if n > 1 else None

                    # Générer le morphing
                    morph = morph_faces(img_a, img_b, alpha=ALPHA, variation_seed=variation_seed)

                    if morph is not None:
                        # Nom du fichier: A_B_N.png
                        filename = f"{name_a}_{name_b}_{n}.png"
                        filepath = OUTPUT_DIR / filename

                        # Sauvegarder
                        cv2.imwrite(str(filepath), morph)

                        pair_success += 1
                        stats["successful"] += 1
                    else:
                        pair_failed += 1
                        stats["failed"] += 1

                except Exception as e:
                    pair_failed += 1
                    stats["failed"] += 1

            # Info de la paire
            stats["pairs_info"].append({
                "pair_index": pair_idx,
                "person_a": person_a,
                "person_b": person_b,
                "successful": pair_success,
                "failed": pair_failed
            })

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
    success_rate = (stats["successful"] / max(1, stats["successful"] + stats["failed"])) * 100

    print(f"""
╔═══════════════════════════════════════════════════════════════════════╗
║                    ✅ GÉNÉRATION TERMINÉE ✅                          ║
╠═══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║  📊 STATISTIQUES:                                                     ║
║                                                                       ║
║     • Paires d'identités: {len(all_pairs):>6}                                    ║
║     • Images par paire:   {NUM_VARIATIONS:>6}                                    ║
║     • Images générées:    {stats["successful"]:>6}                                    ║
║     • Échecs:             {stats["failed"]:>6}                                    ║
║     • Taux de réussite:   {success_rate:>5.1f}%                                   ║
║                                                                       ║
║  ⏱️  TEMPS:                                                           ║
║                                                                       ║
║     • Durée totale:       {elapsed_time/60:>5.1f} minutes                         ║
║     • Vitesse:            {stats["successful"]/max(1,elapsed_time):>5.2f} img/sec                        ║
║                                                                       ║
║  📁 FICHIERS:                                                         ║
║                                                                       ║
║     • Dossier: {str(OUTPUT_DIR):<40} ║
║     • Stats:   dataset_stats.json                                    ║
║                                                                       ║
║  💡 Format: A_B_N.png (A=personne1, B=personne2, N=1-30)             ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝
""")

    print(f"\n✅ Dataset prêt pour la recherche MIA!")
    print(f"📁 Chemin: {OUTPUT_DIR.absolute()}\n")


if __name__ == "__main__":
    main()
