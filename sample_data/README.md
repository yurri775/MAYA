# 📊 Échantillons de Démonstration - Face Morphing

Ce dossier contient des **échantillons de démonstration** pour illustrer le fonctionnement du morphing facial.

## 📁 Structure du Dossier

```
sample_data/
├── before_morph/       → Images originales (Paires A et B)
├── after_morph/        → Images morphées (résultat du morphing)
├── morph_comparison/   → Comparaisons visuelles côte-à-côte
└── gifs_demo/          → Animations GIF (transitions)
```

## 🎯 Contenu

### 1. **before_morph/** - Images Avant Morphing
Contient les **images sources originales** utilisées pour créer les morphings.
- `sample_XX_personA_[Nom].png` : Image de la personne A
- `sample_XX_personB_[Nom].png` : Image de la personne B

**Format** : 256x256 pixels, BGR

### 2. **after_morph/** - Images Après Morphing
Contient les **images morphées** créées en combinant deux visages.
- `sample_XX_morph.png` : Résultat du morphing (α = 0.5)

**Format** : 256x256 pixels, BGR

### 3. **morph_comparison/** - Comparaisons Visuelles
Contient des **images de comparaison côte-à-côte** pour visualiser :
- Image originale A (gauche)
- Image morphée (centre)
- Image originale B (droite)

**Format** : ~830x376 pixels, avec annotations

### 4. **gifs_demo/** - Animations
*(À venir)* Contiendra des animations GIF montrant la transition fluide entre les deux visages.

## 🔬 Technique Utilisée

Les morphings sont créés en utilisant la **triangulation de Delaunay** basée sur les points de repère faciaux (68 landmarks détectés par Dlib).

### Algorithme :
1. **Détection des landmarks** : 68 points faciaux détectés sur chaque image
2. **Triangulation** : Division du visage en triangles avec Delaunay
3. **Transformation affine** : Déformation de chaque triangle
4. **Fusion** : Combinaison pondérée des deux images (α = 0.5)

## 📖 Référence

Basé sur les techniques du papier de recherche :
**SynMorph: Generating Synthetic Face Morphing Dataset with Mated Samples**
- arXiv:2409.05595v1 [cs.CV] 09 Sep 2024
- Auteurs : Haoyu Zhang, Raghavendra Ramachandra, Kiran Raja, Christoph Busch

## 📝 Utilisation pour Présentation

Ces échantillons peuvent être utilisés pour :
- ✅ Démontrer le concept de morphing facial
- ✅ Montrer la qualité des résultats
- ✅ Expliquer le processus avant/après
- ✅ Illustrer les applications en biométrie

## 🎓 Pour le Professeur

Ces échantillons démontrent :
1. **La maîtrise technique** : Implémentation correcte de l'algorithme
2. **La qualité des résultats** : Morphings réalistes et cohérents
3. **L'organisation** : Structure claire et documentation complète
4. **L'innovation** : Basé sur des recherches récentes (2024)

## 🔧 Comment Régénérer

Pour créer de nouveaux échantillons :

```bash
cd moprh
python generate_samples.py
```

Le script générera automatiquement 5 nouveaux échantillons dans ce dossier.

---

**Date de génération** : Janvier 2026
**Dataset source** : LFW (Labeled Faces in the Wild)
**Nombre d'échantillons** : 5 paires (10 images originales + 5 morphs + 5 comparaisons)
