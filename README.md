
# FACEMOMO - Face Morphing Studio

**Génération de morphings faciaux et analyse de vulnérabilité biométrique**

Ce projet propose une solution logicielle pour la création de morphings faciaux de haute qualité. Il intègre des outils d'analyse statistique avancés permettant d'évaluer la qualité des images et leur potentiel d'attaque contre des systèmes de reconnaissance faciale.

## 1. Fonctionnalités

* **Génération de Morphing** : Algorithme basé sur la triangulation de Delaunay pour des transitions fluides.
* **Analyse Statistique (FIQA)** : Évaluation de la qualité des images via trois méthodes biométriques distinctes.
* **Évaluation de Vulnérabilité (MAP)** : Calcul du potentiel d'attaque sur plusieurs systèmes FRS (Face Recognition Systems).
* **Outils de Visualisation** : Génération de courbes DET, distributions KDE et graphiques de comparaison.
* **Suivi de Performance** : Dashboard en temps réel indiquant la vitesse de traitement et les statistiques de succès.



## 2. Fondements Scientifiques

Le projet s'appuie sur les recherches récentes, notamment le papier :
*SynMorph: Generating Synthetic Face Morphing Dataset with Mated Samples (2024)*.
Auteurs : H. Zhang, R. Ramachandra, K. Raja, C. Busch (NTNU / Darmstadt University of Applied Sciences).

## 3. Installation

### Prérequis
Le projet nécessite Python 3.7+ et les bibliothèques suivantes :
```bash
pip install numpy opencv-python dlib matplotlib scikit-learn pillow imageio tqdm seaborn scipy

```

### Modèles

Le modèle de détection des points faciaux (68 landmarks) est automatiquement téléchargé lors du premier lancement du script.

## 4. Utilisation

### Génération via Jupyter Notebook

1. Ouvrez `morph1.ipynb`.
2. Configurez les paramètres (Mode, Nombre d'échantillons, Taille des images).
3. Exécutez les cellules pour lancer la génération et le suivi en temps réel.

### Analyse de Performance

Pour évaluer la qualité et la vulnérabilité de vos résultats :

```bash
python analyze_morphs.py --morph morphing_results --bona-fide sample_data/before_morph

```

Les résultats (Rapports texte et graphiques DET/KDE) sont exportés dans le dossier `statistics_output/`.

## 5. Architecture du Projet

* `morph1.ipynb` : Script principal de génération et interface utilisateur.
* `statistics_module.py` : Moteur de calcul des métriques de sécurité (FIQA, MAP).
* `analyze_morphs.py` : Script d'analyse globale et génération de rapports.
* `sample_data/` : Jeux de données originaux et échantillons de démonstration.

## 6. Dataset

Le projet utilise le dataset **LFW (Labeled Faces in the Wild)** :

* 34 individus sélectionnés.
* Plus de 2300 images disponibles.
* Minimum 30 images par personne pour garantir la pertinence statistique.

## 7. Licence et Auteur

* **Auteur** : AMRANI Ayoub (yurri775),papa abdoulah
  
* **Licence** : Ce projet est sous licence MIT.


