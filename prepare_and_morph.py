"""
Script complet pour:
1. FILTRER les images utilisables (landmarks détectés)
2. ALIGNER les visages (yeux et bouche aux mêmes coordonnées)
3. GÉNÉRER les morphings de haute qualité

Configuration:
- 50 personnes avec le plus d'images valides
- 30 images morphées par paire (format A_B_N.png)
- Alpha fixé à 0.5 (50%)
- Images alignées 256x256
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import cv2
import dlib
from pathlib import Path
import urllib.request
import bz2
from itertools import combinations
from tqdm import tqdm
import time
from datetime import datetime
import json
from collections import defaultdict
import shutil

# ==================== CONFIGURATION ====================

# Source dataset
CASIA_DIR = Path("./casia_webface/CASIA-WebFace_crop")

# Dossiers de sortie
ALIGNED_DIR = Path("./aligned_faces")      # Images alignées filtrées
OUTPUT_DIR = Path("./morphed_dataset")     # Morphings finaux
ALIGNED_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Modèle Dlib
DLIB_DIR = Path("./dlib_models")
DLIB_DIR.mkdir(exist_ok=True)
PREDICTOR_PATH = DLIB_DIR / "shape_predictor_68_face_landmarks.dat"

# Paramètres d'alignement
ALIGNED_SIZE = 256           # Taille des images alignées
LEFT_EYE_POS = (0.35, 0.35)  # Position relative de l'oeil gauche
RIGHT_EYE_POS = (0.65, 0.35) # Position relative de l'oeil droit

# Paramètres de génération
ALPHA = 0.5                  # Mélange fixé à 50%
NUM_VARIATIONS = 30          # 30 images par paire
MAX_PERSONS = 50             # 50 personnes max
MIN_VALID_IMAGES = 10        # Minimum d'images valides par personne

print("="*70)
print("PREPARATION ET GENERATION DE MORPHINGS")
print("="*70)
print(f"""
Configuration:
   - Taille images alignees: {ALIGNED_SIZE}x{ALIGNED_SIZE}
   - Personnes selectionnees: {MAX_PERSONS}
   - Images par paire: {NUM_VARIATIONS}
   - Alpha (melange): {ALPHA} (50%)
   - Format: A_B_N.png
""")

# ==================== TELECHARGER DLIB ====================

def download_dlib():
    if PREDICTOR_PATH.exists():
        print("[OK] Modele Dlib present")
        return

    print("[...] Telechargement du modele Dlib...")
    url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    compressed = DLIB_DIR / "shape_predictor_68_face_landmarks.dat.bz2"

    urllib.request.urlretrieve(url, compressed)
    print("     Decompression...")

    with bz2.BZ2File(compressed, 'rb') as f_in:
        with open(PREDICTOR_PATH, 'wb') as f_out:
            f_out.write(f_in.read())

    compressed.unlink()
    print("[OK] Modele Dlib pret")

download_dlib()

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(str(PREDICTOR_PATH))
print("[OK] Dlib charge")

# ==================== FONCTIONS D'ALIGNEMENT ====================

def get_landmarks(img):
    """Detecte les 68 landmarks faciaux"""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    faces = detector(gray, 1)
    if len(faces) == 0:
        return None

    # Prendre le plus grand visage
    face = max(faces, key=lambda f: (f.right() - f.left()) * (f.bottom() - f.top()))
    shape = predictor(gray, face)

    landmarks = np.zeros((68, 2), dtype=np.float32)
    for i in range(68):
        landmarks[i] = (shape.part(i).x, shape.part(i).y)

    return landmarks

def get_eye_centers(landmarks):
    """Calcule les centres des yeux"""
    # Oeil gauche: points 36-41
    left_eye = landmarks[36:42].mean(axis=0)
    # Oeil droit: points 42-47
    right_eye = landmarks[42:48].mean(axis=0)
    return left_eye, right_eye

def align_face(img, landmarks, output_size=256):
    """
    Aligne le visage pour que les yeux soient toujours aux memes positions
    """
    left_eye, right_eye = get_eye_centers(landmarks)

    # Calculer l'angle de rotation
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))

    # Distance entre les yeux
    dist = np.sqrt(dx**2 + dy**2)

    # Distance desiree entre les yeux dans l'image alignee
    desired_dist = (RIGHT_EYE_POS[0] - LEFT_EYE_POS[0]) * output_size
    scale = desired_dist / dist

    # Centre des yeux
    eyes_center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)

    # Matrice de transformation
    M = cv2.getRotationMatrix2D(eyes_center, angle, scale)

    # Ajuster la translation pour centrer les yeux
    M[0, 2] += (output_size * 0.5 - eyes_center[0])
    M[1, 2] += (output_size * LEFT_EYE_POS[1] - eyes_center[1])

    # Appliquer la transformation
    aligned = cv2.warpAffine(img, M, (output_size, output_size),
                             flags=cv2.INTER_LANCZOS4,
                             borderMode=cv2.BORDER_REPLICATE)

    return aligned

def is_valid_face(landmarks, img_shape):
    """Verifie si le visage est valide (bon cadrage, pas de probleme)"""
    h, w = img_shape[:2]

    # Verifier que tous les landmarks sont dans l'image
    if np.any(landmarks < 0) or np.any(landmarks[:, 0] >= w) or np.any(landmarks[:, 1] >= h):
        return False

    # Verifier la taille du visage (pas trop petit)
    face_width = landmarks[:, 0].max() - landmarks[:, 0].min()
    face_height = landmarks[:, 1].max() - landmarks[:, 1].min()

    if face_width < w * 0.2 or face_height < h * 0.2:
        return False

    # Verifier que les yeux sont detectables
    left_eye, right_eye = get_eye_centers(landmarks)
    eye_dist = np.sqrt((right_eye[0] - left_eye[0])**2 + (right_eye[1] - left_eye[1])**2)

    if eye_dist < 20:  # Trop petit
        return False

    return True

# ==================== ETAPE 1: FILTRER ET ALIGNER ====================

def filter_and_align_dataset():
    """Filtre les images valides et les aligne"""
    print("\n" + "="*70)
    print("ETAPE 1: FILTRAGE ET ALIGNEMENT DES IMAGES")
    print("="*70)

    if not CASIA_DIR.exists():
        print(f"[ERREUR] Dossier non trouve: {CASIA_DIR}")
        return {}

    # Trouver tous les dossiers de personnes
    person_dirs = [d for d in CASIA_DIR.iterdir() if d.is_dir()]
    print(f"\n[INFO] {len(person_dirs)} personnes trouvees dans CASIA-WebFace")

    valid_persons = defaultdict(list)
    total_processed = 0
    total_valid = 0
    total_failed = 0

    for person_dir in tqdm(person_dirs, desc="Traitement des personnes"):
        person_id = person_dir.name

        # Creer le dossier aligne pour cette personne
        aligned_person_dir = ALIGNED_DIR / person_id

        # Collecter les images
        images = list(person_dir.glob("*.jpg")) + list(person_dir.glob("*.png"))

        valid_count = 0

        for img_path in images:
            total_processed += 1

            try:
                # Charger l'image
                img = cv2.imread(str(img_path))
                if img is None:
                    total_failed += 1
                    continue

                # Detecter les landmarks
                landmarks = get_landmarks(img)
                if landmarks is None:
                    total_failed += 1
                    continue

                # Verifier la validite
                if not is_valid_face(landmarks, img.shape):
                    total_failed += 1
                    continue

                # Aligner le visage
                aligned = align_face(img, landmarks, ALIGNED_SIZE)

                # Sauvegarder
                aligned_person_dir.mkdir(exist_ok=True)
                aligned_path = aligned_person_dir / img_path.name
                cv2.imwrite(str(aligned_path), aligned)

                valid_persons[person_id].append(str(aligned_path))
                valid_count += 1
                total_valid += 1

            except Exception as e:
                total_failed += 1
                continue

        # Ne garder que les personnes avec assez d'images valides
        if valid_count < MIN_VALID_IMAGES:
            if person_id in valid_persons:
                del valid_persons[person_id]
            # Supprimer le dossier cree
            if aligned_person_dir.exists():
                shutil.rmtree(aligned_person_dir)

    # Trier par nombre d'images et selectionner les top MAX_PERSONS
    sorted_persons = sorted(valid_persons.items(), key=lambda x: len(x[1]), reverse=True)
    selected = dict(sorted_persons[:MAX_PERSONS])

    print(f"""
[RESULTAT FILTRAGE]
   - Images traitees: {total_processed}
   - Images valides: {total_valid}
   - Images rejetees: {total_failed}
   - Personnes avec {MIN_VALID_IMAGES}+ images: {len(valid_persons)}
   - Personnes selectionnees: {len(selected)}
""")

    return selected

# ==================== ETAPE 2: MORPHING ====================

def add_boundary_points(points, w, h):
    """Ajoute des points aux bords"""
    boundary = np.array([
        [0, 0], [w//2, 0], [w-1, 0],
        [w-1, h//2], [w-1, h-1],
        [w//2, h-1], [0, h-1], [0, h//2]
    ], dtype=np.float32)
    return np.concatenate([points, boundary], axis=0)

def morph_aligned_faces(img_path_a, img_path_b, alpha=0.5):
    """Morphe deux visages alignes"""
    # Charger les images alignees
    imgA = cv2.imread(str(img_path_a))
    imgB = cv2.imread(str(img_path_b))

    if imgA is None or imgB is None:
        return None

    # Les images sont deja alignees a ALIGNED_SIZE x ALIGNED_SIZE
    SIZE = imgA.shape[0]

    # Detecter les landmarks sur les images alignees
    landmarksA = get_landmarks(imgA)
    landmarksB = get_landmarks(imgB)

    # Si landmarks non detectes, utiliser une grille par defaut
    if landmarksA is None:
        x = np.linspace(SIZE*0.2, SIZE*0.8, 8)
        y = np.linspace(SIZE*0.15, SIZE*0.85, 9)
        grid = np.array([(xi, yi) for yi in y for xi in x][:68], dtype=np.float32)
        landmarksA = grid

    if landmarksB is None:
        x = np.linspace(SIZE*0.2, SIZE*0.8, 8)
        y = np.linspace(SIZE*0.15, SIZE*0.85, 9)
        grid = np.array([(xi, yi) for yi in y for xi in x][:68], dtype=np.float32)
        landmarksB = grid

    # Ajouter les points de bordure
    ptsA = add_boundary_points(landmarksA, SIZE, SIZE)
    ptsB = add_boundary_points(landmarksB, SIZE, SIZE)

    # Calculer les points morphes
    pts_morph = (1.0 - alpha) * ptsA + alpha * ptsB

    # Clamper les points
    pts_morph[:, 0] = np.clip(pts_morph[:, 0], 0, SIZE - 1)
    pts_morph[:, 1] = np.clip(pts_morph[:, 1], 0, SIZE - 1)

    # Triangulation de Delaunay
    rect = (0, 0, SIZE, SIZE)
    subdiv = cv2.Subdiv2D(rect)

    for p in pts_morph:
        try:
            subdiv.insert((float(p[0]), float(p[1])))
        except:
            pass

    triangles = subdiv.getTriangleList()

    # Trouver les indices des triangles
    def find_index(points, pt, tol=5.0):
        dists = np.linalg.norm(points - pt, axis=1)
        idx = np.argmin(dists)
        return int(idx) if dists[idx] <= tol else None

    tri_indices = []
    for t in triangles:
        pts = [(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]
        inside = all(0 <= p[0] < SIZE and 0 <= p[1] < SIZE for p in pts)
        if not inside:
            continue

        indices = []
        valid = True
        for pt in pts:
            idx = find_index(pts_morph, np.array(pt))
            if idx is None:
                valid = False
                break
            indices.append(idx)

        if valid and len(set(indices)) == 3:
            tri_indices.append(tuple(indices))

    tri_indices = list(set(tri_indices))

    # Morphing par triangle
    imgA_float = imgA.astype(np.float32)
    imgB_float = imgB.astype(np.float32)
    img_morph = np.zeros_like(imgA_float)

    for tri in tri_indices:
        i1, i2, i3 = tri

        t1 = [ptsA[i1].tolist(), ptsA[i2].tolist(), ptsA[i3].tolist()]
        t2 = [ptsB[i1].tolist(), ptsB[i2].tolist(), ptsB[i3].tolist()]
        tm = [pts_morph[i1].tolist(), pts_morph[i2].tolist(), pts_morph[i3].tolist()]

        # Bounding boxes
        r1 = cv2.boundingRect(np.float32([t1]))
        r2 = cv2.boundingRect(np.float32([t2]))
        rm = cv2.boundingRect(np.float32([tm]))

        if r1[2] <= 0 or r1[3] <= 0 or r2[2] <= 0 or r2[3] <= 0 or rm[2] <= 0 or rm[3] <= 0:
            continue

        # Triangles relatifs
        t1_rect = [(t1[i][0] - r1[0], t1[i][1] - r1[1]) for i in range(3)]
        t2_rect = [(t2[i][0] - r2[0], t2[i][1] - r2[1]) for i in range(3)]
        tm_rect = [(tm[i][0] - rm[0], tm[i][1] - rm[1]) for i in range(3)]

        try:
            # Extraire les regions
            img1_rect = imgA_float[r1[1]:r1[1]+r1[3], r1[0]:r1[0]+r1[2]]
            img2_rect = imgB_float[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]]

            if img1_rect.size == 0 or img2_rect.size == 0:
                continue

            # Transformation affine
            size = (rm[2], rm[3])
            M1 = cv2.getAffineTransform(np.float32(t1_rect), np.float32(tm_rect))
            M2 = cv2.getAffineTransform(np.float32(t2_rect), np.float32(tm_rect))

            warp1 = cv2.warpAffine(img1_rect, M1, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            warp2 = cv2.warpAffine(img2_rect, M2, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

            # Blending
            morphed_rect = (1.0 - alpha) * warp1 + alpha * warp2

            # Masque
            mask = np.zeros((rm[3], rm[2]), dtype=np.float32)
            cv2.fillConvexPoly(mask, np.int32(tm_rect), 1.0, 16, 0)
            mask = mask[:, :, np.newaxis]

            # Appliquer
            y1, y2 = rm[1], rm[1] + rm[3]
            x1, x2 = rm[0], rm[0] + rm[2]

            if y2 <= SIZE and x2 <= SIZE:
                img_morph[y1:y2, x1:x2] = img_morph[y1:y2, x1:x2] * (1 - mask) + morphed_rect * mask
        except:
            continue

    return np.clip(img_morph, 0, 255).astype(np.uint8)

def generate_morphings(persons):
    """Genere tous les morphings"""
    print("\n" + "="*70)
    print("ETAPE 2: GENERATION DES MORPHINGS")
    print("="*70)

    if len(persons) < 2:
        print("[ERREUR] Pas assez de personnes")
        return

    # Generer les paires
    person_ids = list(persons.keys())
    all_pairs = list(combinations(person_ids, 2))

    total = len(all_pairs) * NUM_VARIATIONS

    print(f"""
[INFO] Generation:
   - Paires: {len(all_pairs)}
   - Variations par paire: {NUM_VARIATIONS}
   - Total images: {total}
   - Alpha: {ALPHA}
""")

    stats = {
        "pairs": len(all_pairs),
        "variations": NUM_VARIATIONS,
        "expected": total,
        "successful": 0,
        "failed": 0,
        "start_time": datetime.now().isoformat()
    }

    start_time = time.time()

    with tqdm(total=len(all_pairs), desc="Generation", unit="paire") as pbar:
        for pair_idx, (person_a, person_b) in enumerate(all_pairs):

            # Nettoyer les noms
            name_a = "".join(c if c.isalnum() else '' for c in person_a)[:15]
            name_b = "".join(c if c.isalnum() else '' for c in person_b)[:15]

            imgs_a = persons[person_a]
            imgs_b = persons[person_b]

            # Generer 30 variations
            for n in range(1, NUM_VARIATIONS + 1):
                try:
                    # Selectionner aleatoirement
                    img_a = np.random.choice(imgs_a)
                    img_b = np.random.choice(imgs_b)

                    # Morphing
                    morph = morph_aligned_faces(img_a, img_b, ALPHA)

                    if morph is not None:
                        # Format: A_B_N.png
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
======================================================================
                    GENERATION TERMINEE
======================================================================

   RESULTATS:
      - Images generees: {stats["successful"]}
      - Echecs: {stats["failed"]}
      - Taux reussite: {success_rate:.1f}%

   TEMPS:
      - Duree: {elapsed/60:.1f} minutes
      - Vitesse: {stats["successful"]/max(1,elapsed):.1f} img/sec

   FICHIERS:
      - Dossier: {OUTPUT_DIR}
      - Format: A_B_N.png

======================================================================
""")

# ==================== MAIN ====================

def main():
    print("\n[DEMARRAGE]")

    # Etape 1: Filtrer et aligner
    persons = filter_and_align_dataset()

    if len(persons) < 2:
        print("[ERREUR] Pas assez de personnes valides")
        return

    # Etape 2: Generer les morphings
    generate_morphings(persons)

    print("\n[FIN] Dataset pret!")
    print(f"    Images alignees: {ALIGNED_DIR}")
    print(f"    Morphings: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
