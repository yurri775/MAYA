# 🎭 FACEMOMO - Face Morphing Studio

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Génération artistique de morphings faciaux avec suivi en temps réel**

![Banner](https://img.shields.io/badge/Status-Active-success)

## ✨ Fonctionnalités

- 🎨 **Morphing facial haute qualité** utilisant la triangulation de Delaunay
- 📊 **Dashboard en temps réel** avec statistiques et métriques de performance
- 🎬 **GIFs animés** montrant les transitions fluides entre visages
- 🖼️ **Grilles artistiques** avec design professionnel
- ⚡ **Traitement rapide** avec optimisations OpenCV
- 📈 **Métriques détaillées** : vitesse, temps restant, taux de succès

## 🚀 Installation

### Prérequis

```bash
pip install numpy opencv-python dlib matplotlib scikit-learn pillow imageio tqdm
```

### Téléchargement du modèle Dlib

Le modèle de détection des points faciaux sera téléchargé automatiquement au premier lancement.

## 💻 Utilisation

### Mode Jupyter Notebook

1. Ouvrez `morph1.ipynb` dans Jupyter
2. Exécutez toutes les cellules dans l'ordre
3. Suivez le menu interactif pour configurer la génération
4. Profitez du suivi en temps réel !

### Configuration

```python
MODE = "sample"           # "sample", "per_person", ou "all"
NUM_SAMPLES = 20          # Nombre d'échantillons (mode sample)
ALPHA_VALUES = [0.5]      # Valeurs de morphing (0.0 = image A, 1.0 = image B)
SIZE = 128                # Taille des images générées
CREATE_GIFS = True        # Créer des GIFs animés
CREATE_GRID = True        # Créer une grille artistique
```

## 📁 Structure du Projet

```
moprh/
├── morph1.ipynb              # Notebook principal
├── README.md                 # Ce fichier
├── .gitignore               # Fichiers ignorés par git
├── morphing_results/        # Images générées (non versionnées)
│   ├── gifs/               # GIFs animés
│   └── grids/              # Grilles artistiques
└── dlib_models/            # Modèles de détection (non versionnés)
```

## 🎨 Exemples de Résultats

Le programme génère :
- **Images individuelles** : morphings sauvegardés séparément
- **GIFs animés** : transitions fluides entre visages
- **Grilles artistiques** : compilation esthétique des résultats

## 📊 Métriques en Temps Réel

- ⏱️ Temps écoulé et temps restant estimé
- ⚡ Vitesse de génération (images/seconde)
- 📈 Graphique d'évolution de la vitesse
- ✅ Taux de succès/échec
- 🖼️ Prévisualisation des morphings générés

## 🛠️ Technologies Utilisées

- **Python 3.7+**
- **OpenCV** : Traitement d'images et triangulation
- **Dlib** : Détection des points faciaux (68 landmarks)
- **NumPy** : Calculs numériques
- **Matplotlib** : Visualisations et dashboard
- **scikit-learn** : Dataset LFW (Labeled Faces in the Wild)
- **Pillow & ImageIO** : Création de GIFs animés

## 📝 Modes de Génération

### 1. Sample (Recommandé)
Génère un échantillon aléatoire de morphings
```python
MODE = "sample"
NUM_SAMPLES = 20
```

### 2. Per Person
Un morphing par personne du dataset
```python
MODE = "per_person"
```

### 3. All (Attention !)
Toutes les combinaisons possibles (peut générer des milliers d'images)
```python
MODE = "all"
```

## 🎯 Dataset

Le projet utilise le dataset **LFW (Labeled Faces in the Wild)** :
- 34 personnes
- 2370 images
- Minimum 30 images par personne

## 🤝 Contribution

Les contributions sont les bienvenues ! N'hésitez pas à :
- 🐛 Signaler des bugs
- 💡 Proposer de nouvelles fonctionnalités
- 📝 Améliorer la documentation

## 📜 Licence

Ce projet est sous licence MIT.

## 👨‍💻 Auteur

**Marwa** - [yurri775](https://github.com/yurri775)

## 🙏 Remerciements

- Dataset LFW pour les images de visages
- Bibliothèque Dlib pour la détection des landmarks
- Communauté OpenCV pour les outils de traitement d'images

---

⭐ **Si ce projet vous plaît, n'oubliez pas de mettre une étoile !** ⭐
