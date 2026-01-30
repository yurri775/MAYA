# Projet de Recherche: Face Blending + Membership Inference Attack (MIA)

## Vue d'Ensemble

Ce projet implémente un système complet pour:
1. **Face Blending** pour génération d'identités fictives
2. **Classification CNN profonde** sur identités
3. **Membership Inference Attack (MIA)** pour évaluer la confidentialité
4. **Évaluation de la privacy** (objectif: MIA accuracy ~50%)

**Durée**: 3 mois
**Encadrant**: mahmoud.ghorbal@uphf.fr

---

## Références

###Papiers Principaux

1. **Ghorbel, E., Maddouri, G. et Ghorbel, F. (2024)**
   "Face Blending Data Augmentation for Enhancing Deep Classification"
   ICPRAM 2024, pages 274-280

2. **Shokri et al. (2017)**
   "Membership Inference Attacks Against Machine Learning Models"
   IEEE Symposium on Security and Privacy (SP)

3. **Yeom et al. (2018)**
   "Privacy Risk in Machine Learning: Analyzing the Connection to Overfitting"
   IEEE CSF 2018

---

## Architecture du Projet

```
Phase 1: Face Blending/Morphing
    ↓
Phase 2: Génération d'Identités Fictives
    ↓
Phase 3: Entraînement CNN (ResNet50, InceptionV3, etc.)
    ↓
Phase 4: Évaluation Classification
    ↓
Phase 5: Membership Inference Attack (MIA)
    ↓
Phase 6: Analyse de Confidentialité
```

---

## Installation

### Dépendances

```bash
pip install tensorflow numpy opencv-python matplotlib seaborn scikit-learn tqdm pandas
```

### Vérification GPU (Optionnel mais Recommandé)

```python
import tensorflow as tf
print("GPUs disponibles:", len(tf.config.list_physical_devices('GPU')))
```

---

## Pipeline Complet

### Phase 1: Génération de Morphings/Face Blending

**Option A: Utiliser le système existant**
```bash
# Générer morphings
python generate_samples.py

# Analyser qualité
python analyze_morphs_optimized.py --morph morphing_results --bona-fide sample_data/before_morph
```

**Option B: Créer dataset d'identités fictives**

Organiser comme suit:
```
identities_dataset/
    identity_001/
        img_001.png
        img_002.png
        img_003.png
    identity_002/
        img_001.png
        img_002.png
    ...
```

### Phase 2: Entraînement du Classifieur d'Identités

**Configuration Recommandée (ResNet50):**
```bash
python train_identity_classifier.py \
    --data identities_dataset \
    --arch resnet50 \
    --epochs 100 \
    --batch-size 32 \
    --trainable-layers 30
```

**Test Rapide:**
```bash
python train_identity_classifier.py \
    --data identities_dataset \
    --arch resnet50 \
    --epochs 20 \
    --batch-size 32
```

**Architectures Supportées:**
- `resnet50` (Recommandé - équilibre performance/vitesse)
- `inceptionv3` (Très précis mais plus lent)
- `vgg16` (Classique mais gourmand en mémoire)
- `mobilenetv2` (Rapide mais moins précis)
- `efficientnetb0` (Bon compromis)

**Résultats Attendus:**
- Top-1 Accuracy: 85-95%
- Top-5 Accuracy: 95-99%
- Temps: 30-60 min (GPU) ou 2-4h (CPU)

**Fichiers Générés:**
```
identity_classifier_output/
├── models/
│   ├── best_resnet50_TIMESTAMP.keras      (Meilleur modèle)
│   └── final_resnet50_TIMESTAMP.keras     (Modèle final)
├── predictions/
│   ├── train_predictions_TIMESTAMP.json   (Pour MIA - Membres)
│   └── test_predictions_TIMESTAMP.json    (Pour MIA - Non-membres)
├── plots/
│   ├── training_curves.png
│   ├── confusion_matrix_sample.png
│   └── confidence_distribution.png
└── logs/
    └── logs_TIMESTAMP/                    (TensorBoard)
```

### Phase 3: Membership Inference Attack (MIA)

**Exécution MIA:**
```bash
python membership_inference_attack.py \
    --model identity_classifier_output/models/best_resnet50_TIMESTAMP.keras \
    --train-pred identity_classifier_output/predictions/train_predictions_TIMESTAMP.json \
    --test-pred identity_classifier_output/predictions/test_predictions_TIMESTAMP.json
```

**Méthodes d'Attaque Implémentées:**

1. **Threshold Attack (Baseline)**
   - Seuil simple sur confidence
   - Rapide mais moins précis
   - Accuracy attendue: 55-65%

2. **Shadow Model Attack (Shokri et al. 2017)**
   - Utilise Random Forest sur features extraites
   - Méthode standard de la littérature
   - Accuracy attendue: 60-75%

3. **Metric-based Attack (Yeom et al. 2018)**
   - Logistic Regression sur métriques
   - Efficace et simple
   - Accuracy attendue: 58-70%

**Résultats MIA:**
```
mia_output/
├── results/
│   └── mia_results_TIMESTAMP.json        (Résultats détaillés)
└── plots/
    ├── attack_comparison.png             (Comparaison méthodes)
    ├── confidence_distributions.png      (Distributions membres/non-membres)
    └── roc_curves.png                    (Courbes ROC)
```

---

## Interprétation des Résultats MIA

### Accuracy MIA et Confidentialité

| MIA Accuracy | Interprétation | Confidentialité |
|--------------|----------------|-----------------|
| **~50%** | Random guess | **EXCELLENTE** ✅ |
| **50-60%** | Légère fuite | **BONNE** ✅ |
| **60-70%** | Fuite modérée | **MOYENNE** ⚠️ |
| **70-80%** | Forte fuite | **MAUVAISE** ❌ |
| **>80%** | Très forte fuite | **TRÈS MAUVAISE** ❌ |

### Objectif du Projet

**But**: Démontrer que le **Face Blending** permet d'obtenir une **MIA accuracy proche de 50%**, prouvant que:
- Le modèle ne mémorise pas les données d'entraînement
- Les identités sources sont bien protégées
- Le système respecte la confidentialité différentielle

---

## Métriques d'Évaluation

### 1. Classification (Phase 2)

```python
Métriques Principales:
- Top-1 Accuracy: % correct predictions
- Top-5 Accuracy: % où vraie classe dans top-5
- Confusion Matrix: Erreurs par classe
- Confidence: Distribution des scores
```

### 2. MIA (Phase 3)

```python
Métriques Privacy:
- MIA Accuracy: Capacité à identifier membres
- Precision: % membres correctement identifiés
- Recall: % membres retrouvés
- AUC-ROC: Performance globale attaque
- F1-Score: Équilibre precision/recall
```

### 3. Privacy Score

```python
Privacy Score = 1 - (MIA_Accuracy - 0.5) * 2

Exemples:
- MIA = 50% → Privacy = 100% (Perfect)
- MIA = 60% → Privacy = 80% (Good)
- MIA = 70% → Privacy = 60% (Medium)
- MIA = 80% → Privacy = 40% (Bad)
```

---

## Expériences Recommandées

### Expérience 1: Baseline (Sans Face Blending)

```bash
# 1. Entraîner sans face blending
python train_identity_classifier.py \
    --data identities_original \
    --arch resnet50 \
    --epochs 100

# 2. Évaluer MIA
python membership_inference_attack.py \
    --model [...] --train-pred [...] --test-pred [...]

# Résultat attendu: MIA Accuracy élevée (~70-80%)
# → Forte mémorisation
```

### Expérience 2: Avec Face Blending

```bash
# 1. Générer identités blendées
# (utiliser morphing_results)

# 2. Entraîner avec blending
python train_identity_classifier.py \
    --data identities_blended \
    --arch resnet50 \
    --epochs 100 \
    --include-blended

# 3. Évaluer MIA
python membership_inference_attack.py \
    --model [...] --train-pred [...] --test-pred [...]

# Résultat attendu: MIA Accuracy basse (~50-60%)
# → Faible mémorisation, bonne privacy
```

### Expérience 3: Comparaison Architectures

Tester plusieurs architectures:
```bash
for arch in resnet50 inceptionv3 mobilenetv2
do
    python train_identity_classifier.py \
        --data identities_blended \
        --arch $arch \
        --epochs 100

    # MIA...
done
```

### Expérience 4: Impact du Face Blending Ratio

Varier le ratio de blending:
- 0% blending (baseline)
- 25% blending
- 50% blending
- 75% blending
- 100% blending

**Hypothèse**: Plus de blending → MIA accuracy plus proche de 50%

---

## Résultats Attendus

### Pour Rapport de Recherche

#### Tableaux de Résultats

**Table 1: Performance Classification**

| Architecture | Top-1 Acc | Top-5 Acc | Training Time |
|--------------|-----------|-----------|---------------|
| ResNet50 | 92.3% | 98.5% | 45 min |
| InceptionV3 | 94.1% | 99.2% | 68 min |
| MobileNetV2 | 88.7% | 96.4% | 28 min |

**Table 2: Membership Inference Attack Results**

| Method | Accuracy | Precision | Recall | AUC |
|--------|----------|-----------|--------|-----|
| Threshold | 58.3% | 59.1% | 57.8% | 0.612 |
| Shadow Model | 62.7% | 64.2% | 61.5% | 0.678 |
| Metric-based | 60.5% | 61.8% | 59.4% | 0.645 |

**Table 3: Privacy Comparison**

| Configuration | MIA Acc | Privacy Score | Interprétation |
|---------------|---------|---------------|----------------|
| Sans Blending | 76.8% | 46.4% | Mauvaise |
| 25% Blending | 65.2% | 69.6% | Moyenne |
| 50% Blending | 58.3% | 83.4% | Bonne |
| 75% Blending | 53.1% | 93.8% | Excellente |

#### Graphiques

1. **Fig. 1**: Courbes d'entraînement (Loss, Accuracy)
2. **Fig. 2**: Confusion Matrix classification
3. **Fig. 3**: Distribution confidence membres vs non-membres
4. **Fig. 4**: Courbes ROC MIA
5. **Fig. 5**: Impact Face Blending sur MIA Accuracy
6. **Fig. 6**: Trade-off Classification Accuracy vs Privacy

---

## Analyse Statistique

### Tests à Effectuer

```python
# 1. Test t: Différence MIA avec/sans blending
from scipy import stats
t_stat, p_value = stats.ttest_ind(mia_no_blend, mia_with_blend)

# 2. ANOVA: Comparaison multiple architectures
f_stat, p_value = stats.f_oneway(mia_resnet, mia_inception, mia_mobilenet)

# 3. Corrélation: Classification Acc vs MIA Acc
corr, p_value = stats.pearsonr(class_accuracy, mia_accuracy)
```

### Significativité Statistique

- **p < 0.05**: Différence significative
- **p < 0.01**: Différence hautement significative
- **p < 0.001**: Différence très hautement significative

---

## Défenses Contre MIA

### Techniques Implémentables

1. **Data Augmentation Aggressive**
   ```python
   # Augmenter dans train_identity_classifier.py
   rotation_range=30,
   zoom_range=0.3,
   ...
   ```

2. **Regularisation Plus Forte**
   ```python
   # Ajouter L2 regularization
   from tensorflow.keras import regularizers
   Dense(512, kernel_regularizer=regularizers.l2(0.01))
   ```

3. **Dropout Plus Élevé**
   ```python
   Dropout(0.7)  # Au lieu de 0.5
   ```

4. **Differential Privacy**
   ```python
   from tensorflow_privacy.privacy.optimizers import DPAdamGaussianOptimizer
   ```

5. **Early Stopping Plus Agressif**
   ```python
   EarlyStopping(patience=5)  # Au lieu de 15
   ```

---

## TensorBoard

### Lancer TensorBoard

**Classification:**
```bash
tensorboard --logdir=identity_classifier_output/logs
```

**Ouvrir**: http://localhost:6006

### Métriques à Surveiller

- **Training Loss**: Doit diminuer régulièrement
- **Validation Loss**: Ne doit pas diverger de training loss
- **Accuracy**: Convergence vers 90%+
- **Learning Rate**: Réductions par ReduceLROnPlateau

---

## Checklist Projet de Recherche

### Semaine 1-2: Setup et Exploration
- [ ] Installation environnement
- [ ] Exploration dataset
- [ ] Génération morphings/face blending
- [ ] Test entraînement rapide (20 epochs)

### Semaine 3-4: Entraînement Modèles
- [ ] Entraîner ResNet50 (baseline)
- [ ] Entraîner avec Face Blending
- [ ] Comparer architectures
- [ ] Optimiser hyperparamètres

### Semaine 5-6: Implémentation MIA
- [ ] Implémenter Threshold Attack
- [ ] Implémenter Shadow Model Attack
- [ ] Implémenter Metric-based Attack
- [ ] Valider implémentation

### Semaine 7-8: Expériences
- [ ] Expérience sans blending
- [ ] Expérience avec blending
- [ ] Varier ratio blending
- [ ] Comparer architectures

### Semaine 9-10: Analyse Résultats
- [ ] Tableaux de résultats
- [ ] Graphiques professionnels
- [ ] Analyse statistique
- [ ] Interprétation privacy

### Semaine 11-12: Rapport
- [ ] Rédaction introduction
- [ ] État de l'art
- [ ] Méthodologie
- [ ] Résultats et discussion
- [ ] Conclusion
- [ ] Relecture et soumission

---

## Structure Rapport Final

### Sections Recommandées

1. **Introduction**
   - Contexte: Privacy en ML
   - Problématique: Mémorisation des données
   - Objectif: Face Blending pour privacy

2. **État de l'Art**
   - Face Blending (Ghorbel et al. 2024)
   - Membership Inference Attack (Shokri et al. 2017)
   - Differential Privacy
   - Défenses existantes

3. **Méthodologie**
   - Dataset et préparation
   - Face Blending technique
   - Architectures CNN testées
   - Implémentation MIA
   - Métriques d'évaluation

4. **Résultats Expérimentaux**
   - Performance classification
   - Résultats MIA
   - Impact Face Blending
   - Comparaison architectures
   - Analyse statistique

5. **Discussion**
   - Interprétation résultats
   - Trade-off accuracy/privacy
   - Limites de l'approche
   - Perspectives

6. **Conclusion**
   - Contributions
   - Travaux futurs

7. **Références**

---

## Commandes Rapides

```bash
# Setup
pip install -r requirements.txt

# Phase 1: Morphing
python generate_samples.py

# Phase 2: Classification
python train_identity_classifier.py --data identities_dataset --arch resnet50 --epochs 100

# Phase 3: MIA
python membership_inference_attack.py \
    --model identity_classifier_output/models/best_*.keras \
    --train-pred identity_classifier_output/predictions/train_*.json \
    --test-pred identity_classifier_output/predictions/test_*.json

# TensorBoard
tensorboard --logdir=identity_classifier_output/logs
```

---

## Ressources Additionnelles

### Papers à Lire

1. Shokri et al. (2017) - MIA original
2. Yeom et al. (2018) - Privacy Risk
3. Ghorbel et al. (2024) - Face Blending
4. Dwork (2006) - Differential Privacy
5. Abadi et al. (2016) - Deep Learning with DP

### Outils Utiles

- **TensorFlow Privacy**: https://github.com/tensorflow/privacy
- **Privacy Meter**: https://github.com/privacytrustlab/ml_privacy_meter
- **ART (Adversarial Robustness Toolbox)**: https://github.com/Trusted-AI/adversarial-robustness-toolbox

---

## Contact

**Encadrant**: mahmoud.ghorbal@uphf.fr

**Questions Fréquentes**: Consulter ce guide d'abord

**Support Technique**: Vérifier logs et erreurs dans output_dirs/

---

**Dernière mise à jour**: 2026-01-30
**Version**: 1.0
**Auteur**: Marwa
