"""
Script avancé pour générer des morphings sur la base de données LFW
Version améliorée avec suivi détaillé, logging et statistiques
"""

import numpy as np
import cv2
import dlib
import matplotlib.pyplot as plt
import os
from pathlib import Path
import urllib.request
import bz2
from sklearn.datasets import fetch_lfw_people
from itertools import combinations
from tqdm import tqdm
import time
import json
import logging
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict
from datetime import datetime
import pandas as pd

# ==================== Configuration ====================

@dataclass
class MorphingConfig:
    """Configuration pour la génération de morphings"""
    # Paramètres du dataset
    min_faces_per_person: int = 30
    resize_factor: float = 0.5
    image_size: int = 128

    # Mode de génération
    mode: str = "sample"  # "all", "sample", "per_person"
    num_samples: int = 50
    alpha_values: List[float] = None

    # Options de sauvegarde
    save_individual: bool = True
    create_grid: bool = True
    grid_size: Tuple[int, int] = (5, 5)
    save_html_report: bool = True

    # Chemins
    output_dir: Path = Path("./morphing_results")
    log_dir: Path = Path("./morphing_logs")
    dlib_models_dir: Path = Path("./dlib_models")

    def __post_init__(self):
        if self.alpha_values is None:
            self.alpha_values = [0.3, 0.5, 0.7]

        # Créer les dossiers
        self.output_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)
        self.dlib_models_dir.mkdir(exist_ok=True)


@dataclass
class MorphingResult:
    """Résultat d'un morphing"""
    person_a: str
    person_b: str
    alpha: float
    index: int
    success: bool
    error_message: Optional[str] = None
    processing_time: float = 0.0
    landmarks_detected_a: bool = True
    landmarks_detected_b: bool = True
    file_path: Optional[str] = None


class ProgressTracker:
    """Suivi détaillé de la progression"""

    def __init__(self, total_tasks: int, log_dir: Path):
        self.total_tasks = total_tasks
        self.completed = 0
        self.successful = 0
        self.failed = 0
        self.start_time = time.time()
        self.log_file = log_dir / f"progress_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.results: List[MorphingResult] = []

    def update(self, result: MorphingResult):
        """Met à jour la progression"""
        self.completed += 1
        if result.success:
            self.successful += 1
        else:
            self.failed += 1
        self.results.append(result)

    def get_stats(self) -> Dict:
        """Retourne les statistiques actuelles"""
        elapsed = time.time() - self.start_time
        remaining = self.total_tasks - self.completed
        speed = self.completed / elapsed if elapsed > 0 else 0
        eta = remaining / speed if speed > 0 else 0

        return {
            "total": self.total_tasks,
            "completed": self.completed,
            "successful": self.successful,
            "failed": self.failed,
            "progress_percent": (self.completed / self.total_tasks * 100) if self.total_tasks > 0 else 0,
            "elapsed_time": elapsed,
            "speed_per_sec": speed,
            "eta_seconds": eta,
            "success_rate": (self.successful / self.completed * 100) if self.completed > 0 else 0
        }

    def save_progress(self):
        """Sauvegarde la progression dans un fichier"""
        stats = self.get_stats()
        stats["timestamp"] = datetime.now().isoformat()
        stats["results"] = [asdict(r) for r in self.results]

        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

    def print_summary(self):
        """Affiche un résumé détaillé"""
        stats = self.get_stats()

        print(f"\n{'='*70}")
        print(f"📊 STATISTIQUES FINALES")
        print(f"{'='*70}")
        print(f"\n✅ Morphings réussis: {self.successful}/{self.total_tasks} ({stats['success_rate']:.1f}%)")
        print(f"❌ Morphings échoués: {self.failed}")
        print(f"⏱️  Temps total: {stats['elapsed_time']:.1f}s ({stats['elapsed_time']/60:.1f} min)")
        print(f"⚡ Vitesse moyenne: {stats['speed_per_sec']:.2f} morphings/sec")

        if self.failed > 0:
            print(f"\n⚠️  Erreurs rencontrées:")
            error_types = {}
            for result in self.results:
                if not result.success and result.error_message:
                    error_types[result.error_message] = error_types.get(result.error_message, 0) + 1

            for error, count in error_types.items():
                print(f"   - {error}: {count} fois")


# ==================== Logging Setup ====================

def setup_logging(log_dir: Path) -> logging.Logger:
    """Configure le système de logging"""
    log_file = log_dir / f"morphing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger(__name__)


# ==================== Morphing Engine ====================

class MorphingEngine:
    """Moteur de génération de morphings"""

    def __init__(self, config: MorphingConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.detector = None
        self.predictor = None
        self.images = None
        self.labels = None
        self.target_names = None

    def initialize(self):
        """Initialise le moteur"""
        self.logger.info("Initialisation du moteur de morphing...")

        # Télécharger et charger Dlib
        predictor_path = self._download_dlib_predictor()
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(str(predictor_path))
        self.logger.info("✓ Modèle Dlib chargé")

        # Charger le dataset
        self._load_dataset()
        self.logger.info("✓ Dataset LFW chargé")

    def _download_dlib_predictor(self) -> Path:
        """Télécharge le modèle Dlib si nécessaire"""
        predictor_path = self.config.dlib_models_dir / "shape_predictor_68_face_landmarks.dat"

        if predictor_path.exists():
            self.logger.info("Modèle Dlib déjà présent")
            return predictor_path

        self.logger.info("Téléchargement du modèle Dlib...")
        url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
        compressed_file = self.config.dlib_models_dir / "shape_predictor_68_face_landmarks.dat.bz2"

        urllib.request.urlretrieve(url, compressed_file)
        self.logger.info("Décompression...")

        with bz2.BZ2File(compressed_file, 'rb') as f_in:
            with open(predictor_path, 'wb') as f_out:
                f_out.write(f_in.read())

        compressed_file.unlink()
        self.logger.info("✓ Modèle Dlib prêt")

        return predictor_path

    def _load_dataset(self):
        """Charge le dataset LFW"""
        self.logger.info("Chargement du dataset LFW...")

        lfw_people = fetch_lfw_people(
            min_faces_per_person=self.config.min_faces_per_person,
            resize=self.config.resize_factor,
            color=False
        )

        self.images = lfw_people.images
        self.labels = lfw_people.target
        self.target_names = lfw_people.target_names

        n_samples, h, w = self.images.shape
        n_classes = len(self.target_names)

        self.logger.info(f"Dataset: {n_samples} images, {n_classes} personnes, taille {h}x{w}")

    def get_landmarks(self, img_gray: np.ndarray) -> Optional[np.ndarray]:
        """Détecte les landmarks faciaux"""
        dets = self.detector(img_gray, 0)
        if len(dets) == 0:
            return None

        shape = self.predictor(img_gray, dets[0])
        pts = np.zeros((68, 2), dtype=np.int32)
        for i in range(68):
            pts[i] = (shape.part(i).x, shape.part(i).y)
        return pts

    def _add_corner_points(self, points: np.ndarray, w: int, h: int) -> np.ndarray:
        """Ajoute les points de coin pour la triangulation"""
        corners = np.array([
            [0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1],
            [w // 2, 0], [w - 1, h // 2], [w // 2, h - 1], [0, h // 2]
        ], dtype=np.int32)
        return np.concatenate([points, corners], axis=0)

    def _clamp_points(self, points: np.ndarray, w: int, h: int) -> np.ndarray:
        """Limite les points aux dimensions de l'image"""
        pts = np.array(points, dtype=np.float32)
        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
        return pts

    def _prepare_points(self, img_gray: np.ndarray, w: int, h: int) -> Tuple[np.ndarray, bool]:
        """Prépare les points pour le morphing"""
        pts = self.get_landmarks(img_gray)
        landmarks_detected = pts is not None

        if pts is None:
            # Créer une grille par défaut
            grid_x = np.tile(np.linspace(w*0.25, w*0.75, 17), (4,))
            grid_y = np.repeat(np.linspace(h*0.25, h*0.75, 4), 17)
            pts = np.vstack([grid_x[:68], grid_y[:68]]).T.astype(np.int32)

        pts = self._clamp_points(pts, w, h)
        pts = self._add_corner_points(pts.astype(np.int32), w, h)
        return pts.astype(np.float32), landmarks_detected

    def _apply_affine_transform(self, src: np.ndarray, src_tri: List, dst_tri: List, size: Tuple) -> np.ndarray:
        """Applique une transformation affine"""
        warp_mat = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))
        dst = cv2.warpAffine(src, warp_mat, (int(size[0]), int(size[1])),
                           None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        return dst

    def _morph_triangle(self, img1: np.ndarray, img2: np.ndarray, img_morphed: np.ndarray,
                       t1: List, t2: List, t_morphed: List, alpha: float):
        """Morphe un triangle"""
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
        warp_img1 = self._apply_affine_transform(img1_rect, t1_rect, t_rect, size_rect)
        warp_img2 = self._apply_affine_transform(img2_rect, t2_rect, t_rect, size_rect)

        img_rect = (1.0 - alpha) * warp_img1 + alpha * warp_img2

        mask = np.zeros((r[3], r[2]), dtype=np.float32)
        cv2.fillConvexPoly(mask, np.int32(t_rect), 1.0, 16, 0)

        y, x, w_rect, h_rect = r[1], r[0], r[2], r[3]
        img_morphed[y:y+h_rect, x:x+w_rect] = (
            img_morphed[y:y+h_rect, x:x+w_rect] * (1 - mask[:, :, None]) +
            img_rect * mask[:, :, None]
        )

    def morph_faces(self, imgA: np.ndarray, imgB: np.ndarray, alpha: float = 0.5
                   ) -> Tuple[Optional[np.ndarray], bool, bool]:
        """Morphe deux visages"""
        size = self.config.image_size

        # Redimensionner
        imgA_resized = cv2.resize(imgA, (size, size), interpolation=cv2.INTER_CUBIC)
        imgB_resized = cv2.resize(imgB, (size, size), interpolation=cv2.INTER_CUBIC)

        # Convertir en uint8
        if imgA_resized.dtype != np.uint8:
            imgA_resized = (imgA_resized * 255).astype(np.uint8)
        if imgB_resized.dtype != np.uint8:
            imgB_resized = (imgB_resized * 255).astype(np.uint8)

        # Convertir en couleur
        imgA_color = cv2.cvtColor(imgA_resized, cv2.COLOR_GRAY2BGR).astype(np.float32)
        imgB_color = cv2.cvtColor(imgB_resized, cv2.COLOR_GRAY2BGR).astype(np.float32)

        # Préparer les points
        ptsA, landmarks_a = self._prepare_points(imgA_resized, size, size)
        ptsB, landmarks_b = self._prepare_points(imgB_resized, size, size)

        # Points morphés
        points_morphed = (1.0 - alpha) * ptsA + alpha * ptsB
        points_morphed = self._clamp_points(points_morphed, size, size)

        # Triangulation de Delaunay
        rect = (0, 0, size, size)
        subdiv = cv2.Subdiv2D(rect)

        for p in points_morphed:
            x, y = float(p[0]), float(p[1])
            if 0 <= x < size and 0 <= y < size:
                subdiv.insert((x, y))

        triangle_list = subdiv.getTriangleList()

        # Trouver les indices des triangles
        tri_indices = []
        for t in triangle_list:
            tri_pts = [(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]
            inds = []
            valid = True
            for p in tri_pts:
                idx = self._find_point_index(points_morphed, p)
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

            if self._triangle_inside(tA, size, size) and \
               self._triangle_inside(tB, size, size) and \
               self._triangle_inside(tM, size, size):
                self._morph_triangle(imgA_color, imgB_color, img_morphed, tA, tB, tM, alpha)

        return np.clip(img_morphed, 0, 255).astype(np.uint8), landmarks_a, landmarks_b

    def _find_point_index(self, points: np.ndarray, pt: Tuple, tol: float = 5.0) -> Optional[int]:
        """Trouve l'index d'un point"""
        pts = np.asarray(points, dtype=np.float32)
        dists = np.linalg.norm(pts - np.asarray(pt, dtype=np.float32), axis=1)
        idx = int(np.argmin(dists))
        return idx if dists[idx] <= tol else None

    def _triangle_inside(self, t: List, w: int, h: int) -> bool:
        """Vérifie si un triangle est complètement dans l'image"""
        for (x, y) in t:
            if x < 0 or x >= w or y < 0 or y >= h:
                return False
        return True

    def generate_person_pairs(self) -> List[Tuple[int, int]]:
        """Génère les paires de personnes selon le mode"""
        unique_labels = np.unique(self.labels)
        n_people = len(unique_labels)

        if self.config.mode == "all":
            pairs = list(combinations(unique_labels, 2))
        elif self.config.mode == "sample":
            all_pairs = list(combinations(unique_labels, 2))
            n_samples = min(self.config.num_samples, len(all_pairs))
            indices = np.random.choice(len(all_pairs), n_samples, replace=False)
            pairs = [all_pairs[i] for i in indices]
        elif self.config.mode == "per_person":
            pairs = []
            for i, person_a in enumerate(unique_labels):
                person_b = unique_labels[(i + 1) % n_people]
                pairs.append((person_a, person_b))
        else:
            raise ValueError(f"Mode inconnu: {self.config.mode}")

        return pairs

    def run(self) -> ProgressTracker:
        """Lance la génération de morphings"""
        pairs = self.generate_person_pairs()
        total_tasks = len(pairs) * len(self.config.alpha_values)

        self.logger.info(f"Génération de {total_tasks} morphings ({len(pairs)} paires, {len(self.config.alpha_values)} alphas)")

        tracker = ProgressTracker(total_tasks, self.config.log_dir)

        with tqdm(total=total_tasks, desc="Progression globale") as pbar:
            for idx, (person_a, person_b) in enumerate(pairs):
                # Récupérer les images
                imgs_a = self.images[self.labels == person_a]
                imgs_b = self.images[self.labels == person_b]

                imgA = imgs_a[np.random.randint(len(imgs_a))]
                imgB = imgs_b[np.random.randint(len(imgs_b))]

                name_a = self.target_names[person_a]
                name_b = self.target_names[person_b]

                for alpha in self.config.alpha_values:
                    start_time = time.time()

                    try:
                        img_morphed, landmarks_a, landmarks_b = self.morph_faces(imgA, imgB, alpha=alpha)

                        # Sauvegarder
                        file_path = None
                        if self.config.save_individual:
                            filename = f"morph_{idx:04d}_{name_a[:10]}_{name_b[:10]}_alpha{alpha:.2f}.png"
                            file_path = self.config.output_dir / filename
                            cv2.imwrite(str(file_path), img_morphed)

                        result = MorphingResult(
                            person_a=name_a,
                            person_b=name_b,
                            alpha=alpha,
                            index=idx,
                            success=True,
                            processing_time=time.time() - start_time,
                            landmarks_detected_a=landmarks_a,
                            landmarks_detected_b=landmarks_b,
                            file_path=str(file_path) if file_path else None
                        )

                    except Exception as e:
                        result = MorphingResult(
                            person_a=name_a,
                            person_b=name_b,
                            alpha=alpha,
                            index=idx,
                            success=False,
                            error_message=str(e),
                            processing_time=time.time() - start_time
                        )
                        self.logger.error(f"Erreur sur {name_a} + {name_b}: {e}")

                    tracker.update(result)
                    pbar.update(1)

                    # Sauvegarder la progression tous les 10 morphings
                    if tracker.completed % 10 == 0:
                        tracker.save_progress()

        # Sauvegarde finale
        tracker.save_progress()
        return tracker


# ==================== Report Generator ====================

class ReportGenerator:
    """Générateur de rapports HTML"""

    def __init__(self, config: MorphingConfig, tracker: ProgressTracker):
        self.config = config
        self.tracker = tracker

    def generate_html_report(self):
        """Génère un rapport HTML interactif"""
        html_file = self.config.output_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"

        stats = self.tracker.get_stats()

        html_content = f"""
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Rapport de Morphing - {datetime.now().strftime('%Y-%m-%d %H:%M')}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }}
        .container {{
            background: white;
            border-radius: 10px;
            padding: 30px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        }}
        h1 {{
            color: #667eea;
            text-align: center;
            margin-bottom: 30px;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }}
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            margin: 10px 0;
        }}
        .stat-label {{
            font-size: 0.9em;
            opacity: 0.9;
        }}
        .gallery {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 30px;
        }}
        .gallery-item {{
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            overflow: hidden;
            transition: transform 0.3s;
        }}
        .gallery-item:hover {{
            transform: scale(1.05);
            box-shadow: 0 5px 20px rgba(0,0,0,0.2);
        }}
        .gallery-item img {{
            width: 100%;
            height: auto;
            display: block;
        }}
        .gallery-item .caption {{
            padding: 10px;
            background: #f5f5f5;
            font-size: 0.8em;
            text-align: center;
        }}
        .progress-bar {{
            width: 100%;
            height: 30px;
            background: #e0e0e0;
            border-radius: 15px;
            overflow: hidden;
            margin: 20px 0;
        }}
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
            transition: width 0.3s;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #e0e0e0;
        }}
        th {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }}
        tr:hover {{
            background: #f5f5f5;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Rapport de Génération de Morphings</h1>

        <div class="stats">
            <div class="stat-card">
                <div class="stat-label">Total Morphings</div>
                <div class="stat-value">{self.tracker.total_tasks}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Réussis</div>
                <div class="stat-value">{self.tracker.successful}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Échoués</div>
                <div class="stat-value">{self.tracker.failed}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Taux de Réussite</div>
                <div class="stat-value">{stats['success_rate']:.1f}%</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Temps Total</div>
                <div class="stat-value">{stats['elapsed_time']/60:.1f} min</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Vitesse</div>
                <div class="stat-value">{stats['speed_per_sec']:.2f}/s</div>
            </div>
        </div>

        <div class="progress-bar">
            <div class="progress-fill" style="width: {stats['progress_percent']}%">
                {stats['progress_percent']:.1f}%
            </div>
        </div>

        <h2>📸 Galerie des Morphings</h2>
        <div class="gallery">
"""

        # Ajouter les images réussies
        for result in self.tracker.results:
            if result.success and result.file_path:
                filename = Path(result.file_path).name
                html_content += f"""
            <div class="gallery-item">
                <img src="{filename}" alt="{result.person_a} + {result.person_b}">
                <div class="caption">
                    {result.person_a[:15]} + {result.person_b[:15]}<br>
                    α = {result.alpha:.2f}
                </div>
            </div>
"""

        html_content += """
        </div>

        <h2>📋 Détails des Morphings</h2>
        <table>
            <tr>
                <th>Index</th>
                <th>Personne A</th>
                <th>Personne B</th>
                <th>Alpha</th>
                <th>Temps (s)</th>
                <th>Statut</th>
            </tr>
"""

        for result in self.tracker.results:
            status = "✅ Réussi" if result.success else f"❌ {result.error_message}"
            html_content += f"""
            <tr>
                <td>{result.index}</td>
                <td>{result.person_a[:20]}</td>
                <td>{result.person_b[:20]}</td>
                <td>{result.alpha:.2f}</td>
                <td>{result.processing_time:.3f}</td>
                <td>{status}</td>
            </tr>
"""

        html_content += """
        </table>

        <p style="text-align: center; color: #999; margin-top: 40px;">
            Généré le """ + datetime.now().strftime('%Y-%m-%d à %H:%M:%S') + """
        </p>
    </div>
</body>
</html>
"""

        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return html_file

    def generate_csv_report(self):
        """Génère un rapport CSV avec toutes les données"""
        csv_file = self.config.output_dir / f"data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        df = pd.DataFrame([asdict(r) for r in self.tracker.results])
        df.to_csv(csv_file, index=False, encoding='utf-8')

        return csv_file


# ==================== Main ====================

def main():
    print(f"\n{'='*70}")
    print(f"🎭 GÉNÉRATEUR DE MORPHINGS AVANCÉ")
    print(f"{'='*70}\n")

    # Configuration
    config = MorphingConfig(
        mode="sample",
        num_samples=25,
        alpha_values=[0.3, 0.5, 0.7],
        save_individual=True,
        create_grid=False,
        save_html_report=True
    )

    # Setup logging
    logger = setup_logging(config.log_dir)
    logger.info("Démarrage de la génération de morphings")

    # Initialiser le moteur
    engine = MorphingEngine(config, logger)
    engine.initialize()

    # Générer les morphings
    tracker = engine.run()

    # Afficher le résumé
    tracker.print_summary()

    # Générer les rapports
    if config.save_html_report:
        report_gen = ReportGenerator(config, tracker)
        html_file = report_gen.generate_html_report()
        csv_file = report_gen.generate_csv_report()

        print(f"\n📄 Rapports générés:")
        print(f"   - HTML: {html_file}")
        print(f"   - CSV: {csv_file}")
        print(f"   - Logs: {tracker.log_file}")

    print(f"\n{'='*70}")
    print(f"✅ GÉNÉRATION TERMINÉE!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
