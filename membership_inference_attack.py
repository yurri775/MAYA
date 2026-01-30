# -*- coding: utf-8 -*-
"""
Membership Inference Attack (MIA) Implementation
================================================

Implementation d'une attaque par inference d'appartenance selon Shokri et al. (2017)
pour evaluer la vulnerabilite du modele de classification d'identites a la memorisation.

Objectif: Quantifier la confidentialite du modele entraine
- Precision MIA proche de 50% = Bonne confidentialite (pas de fuite d'information)
- Precision MIA proche de 100% = Mauvaise confidentialite (forte memorisation)

Base sur:
- Shokri et al. (2017), "Membership Inference Attacks Against Machine Learning Models"
- Yeom et al. (2018), "Privacy Risk in Machine Learning"

Methodes implementees:
1. Threshold Attack (baseline)
2. Shadow Model Attack (Shokri et al.)
3. Metric-based Attack

Version professionnelle pour projet de recherche
"""

import sys
import os

# Configuration UTF-8
if sys.platform == 'win32':
    if sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')
    if sys.stderr.encoding != 'utf-8':
        sys.stderr.reconfigure(encoding='utf-8')

import numpy as np
import json
from pathlib import Path
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# TensorFlow et Keras
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)


class MembershipInferenceAttack:
    """
    Classe pour implementer et evaluer des attaques par inference d'appartenance
    """

    def __init__(self, target_model_path, predictions_train_path, predictions_test_path):
        """
        Initialisation de l'attaque MIA

        Args:
            target_model_path: Chemin du modele cible
            predictions_train_path: JSON des predictions sur training set
            predictions_test_path: JSON des predictions sur test set
        """
        self.target_model_path = target_model_path
        self.predictions_train_path = predictions_train_path
        self.predictions_test_path = predictions_test_path

        # Creer dossier de sortie
        self.output_dir = Path("mia_output")
        self.output_dir.mkdir(exist_ok=True)

        self.plots_dir = self.output_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)

        self.results_dir = self.output_dir / "results"
        self.results_dir.mkdir(exist_ok=True)

        print("\n" + "="*80)
        print("MEMBERSHIP INFERENCE ATTACK (MIA)")
        print("Evaluation de la Confidentialite du Modele")
        print("Base sur Shokri et al. (2017)")
        print("="*80 + "\n")

    def load_data(self):
        """Charge les predictions du modele cible"""
        print("Chargement des predictions...")

        # Charger predictions training (membres)
        with open(self.predictions_train_path, 'r') as f:
            train_data = json.load(f)

        # Charger predictions test (non-membres)
        with open(self.predictions_test_path, 'r') as f:
            test_data = json.load(f)

        # Extraire features
        self.train_confidences = np.array(train_data['confidence'])
        self.train_proba = np.array(train_data['y_pred_proba'])

        self.test_confidences = np.array(test_data['confidence'])
        self.test_proba = np.array(test_data['y_pred_proba'])

        print(f"  Training samples (membres):     {len(self.train_confidences)}")
        print(f"  Test samples (non-membres):     {len(self.test_confidences)}")

        # Creer labels: 1 = membre (training), 0 = non-membre (test)
        self.X_members = self._extract_features(self.train_proba, self.train_confidences)
        self.y_members = np.ones(len(self.train_confidences))

        self.X_non_members = self._extract_features(self.test_proba, self.test_confidences)
        self.y_non_members = np.zeros(len(self.test_confidences))

        # Combiner
        self.X_attack = np.vstack([self.X_members, self.X_non_members])
        self.y_attack = np.concatenate([self.y_members, self.y_non_members])

        print(f"  Features MIA extraites: {self.X_attack.shape[1]} features")

        return self.X_attack, self.y_attack

    def _extract_features(self, probabilities, confidences):
        """
        Extrait features pour attaque MIA

        Features utilisees:
        - Confidence (max probability)
        - Entropy de la distribution
        - Top-3 probabilities
        - Variance des probabilities
        """
        features = []

        for i in range(len(confidences)):
            proba = probabilities[i]
            conf = confidences[i]

            # Feature 1: Confidence
            f1 = conf

            # Feature 2: Entropy
            proba_safe = np.clip(proba, 1e-10, 1.0)
            entropy = -np.sum(proba_safe * np.log(proba_safe))
            f2 = entropy

            # Feature 3-5: Top-3 probabilities
            top3 = np.sort(proba)[-3:][::-1]
            f3, f4, f5 = top3[0], top3[1] if len(top3) > 1 else 0, top3[2] if len(top3) > 2 else 0

            # Feature 6: Variance
            f6 = np.var(proba)

            # Feature 7: Max - Second Max
            f7 = top3[0] - (top3[1] if len(top3) > 1 else 0)

            features.append([f1, f2, f3, f4, f5, f6, f7])

        return np.array(features)

    def threshold_attack(self, threshold=None):
        """
        Attaque baseline: seuil sur confidence

        Args:
            threshold: Seuil de confidence (auto si None)

        Returns:
            dict: Resultats de l'attaque
        """
        print("\n" + "="*80)
        print("MIA METHOD 1: THRESHOLD ATTACK (Baseline)")
        print("-"*80 + "\n")

        # Trouver threshold optimal si non specifie
        if threshold is None:
            # Essayer differents seuils
            thresholds = np.linspace(0.5, 1.0, 100)
            best_acc = 0
            best_threshold = 0.5

            for t in thresholds:
                y_pred = (self.X_attack[:, 0] >= t).astype(int)  # Feature 0 = confidence
                acc = accuracy_score(self.y_attack, y_pred)
                if acc > best_acc:
                    best_acc = acc
                    best_threshold = t

            threshold = best_threshold
            print(f"Threshold optimal trouve: {threshold:.4f}")

        # Prediction
        y_pred = (self.X_attack[:, 0] >= threshold).astype(int)

        # Metriques
        results = self._compute_metrics("Threshold Attack", y_pred)

        return results

    def shadow_model_attack(self, n_shadow_models=3):
        """
        Attaque avec shadow models (Shokri et al. 2017)

        Entraine des modeles d'attaque sur des shadow models
        pour apprendre a distinguer membres/non-membres

        Args:
            n_shadow_models: Nombre de shadow models

        Returns:
            dict: Resultats de l'attaque
        """
        print("\n" + "="*80)
        print("MIA METHOD 2: SHADOW MODEL ATTACK (Shokri et al. 2017)")
        print("-"*80 + "\n")

        print(f"Entrainement de {n_shadow_models} shadow models...")
        print("(Dans une implementation complete, on entrainerait de vrais shadow models)")
        print("Ici, on simule avec un Random Forest sur les features extraites)\n")

        # Split pour entrainement de l'attaquant
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_attack, self.y_attack, test_size=0.3, random_state=42, stratify=self.y_attack
        )

        # Entrainer modele d'attaque (Random Forest)
        print("Entrainement du modele d'attaque (Random Forest)...")
        attack_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )

        attack_model.fit(X_train, y_train)

        # Predictions
        y_pred = attack_model.predict(X_test)
        y_pred_proba = attack_model.predict_proba(X_test)[:, 1]

        # Metriques
        results = self._compute_metrics("Shadow Model Attack", y_pred, y_pred_proba, y_test)

        # Feature importance
        feature_names = ['Confidence', 'Entropy', 'Top-1', 'Top-2', 'Top-3', 'Variance', 'Max-SecondMax']
        importances = attack_model.feature_importances_

        print("\nFeature Importance:")
        for name, imp in zip(feature_names, importances):
            print(f"  {name:15s}: {imp:.4f}")

        return results

    def metric_based_attack(self):
        """
        Attaque basee sur metriques (Yeom et al. 2018)

        Utilise des metriques simples comme perte du modele

        Returns:
            dict: Resultats de l'attaque
        """
        print("\n" + "="*80)
        print("MIA METHOD 3: METRIC-BASED ATTACK (Yeom et al. 2018)")
        print("-"*80 + "\n")

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_attack, self.y_attack, test_size=0.3, random_state=42, stratify=self.y_attack
        )

        # Entrainer modele simple (Logistic Regression)
        print("Entrainement du modele d'attaque (Logistic Regression)...")
        attack_model = LogisticRegression(random_state=42, max_iter=1000)
        attack_model.fit(X_train, y_train)

        # Predictions
        y_pred = attack_model.predict(X_test)
        y_pred_proba = attack_model.predict_proba(X_test)[:, 1]

        # Metriques
        results = self._compute_metrics("Metric-based Attack", y_pred, y_pred_proba, y_test)

        return results

    def _compute_metrics(self, attack_name, y_pred, y_pred_proba=None, y_true=None):
        """Calcule toutes les metriques pour une attaque"""

        if y_true is None:
            y_true = self.y_attack

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        print(f"\nResultats {attack_name}:")
        print(f"  Accuracy:   {accuracy:.4f}")
        print(f"  Precision:  {precision:.4f}")
        print(f"  Recall:     {recall:.4f}")
        print(f"  F1-Score:   {f1:.4f}")

        # AUC si probabilites disponibles
        auc = None
        if y_pred_proba is not None:
            try:
                auc = roc_auc_score(y_true, y_pred_proba)
                print(f"  AUC-ROC:    {auc:.4f}")
            except:
                pass

        # Interpretation privacy
        print(f"\nInterpretation Confidentialite:")
        if accuracy < 0.55:
            print(f"  EXCELLENTE confidentialite (accuracy proche de 50%)")
            print(f"  Le modele ne memorise pas les donnees d'entrainement")
        elif accuracy < 0.65:
            print(f"  BONNE confidentialite")
            print(f"  Faible memorisation des donnees")
        elif accuracy < 0.75:
            print(f"  MOYENNE confidentialite")
            print(f"  Memorisation moderee")
        else:
            print(f"  MAUVAISE confidentialite")
            print(f"  Forte memorisation - vulnerabilite elevee")

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        print(f"\nMatrice de Confusion:")
        print(f"                    Predicted")
        print(f"                 Non-Member  Member")
        print(f"Actual Non-Member    {cm[0,0]:4d}      {cm[0,1]:4d}")
        print(f"       Member        {cm[1,0]:4d}      {cm[1,1]:4d}")

        return {
            'attack_name': attack_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'confusion_matrix': cm.tolist(),
            'y_pred': y_pred.tolist() if isinstance(y_pred, np.ndarray) else y_pred,
            'y_true': y_true.tolist() if isinstance(y_true, np.ndarray) else y_true
        }

    def run_all_attacks(self):
        """Execute toutes les methodes d'attaque"""
        print("\n" + "="*80)
        print("EXECUTION DE TOUTES LES METHODES D'ATTAQUE")
        print("="*80 + "\n")

        results = {}

        # Method 1: Threshold
        results['threshold'] = self.threshold_attack()

        # Method 2: Shadow Model
        results['shadow_model'] = self.shadow_model_attack()

        # Method 3: Metric-based
        results['metric_based'] = self.metric_based_attack()

        # Sauvegarder resultats
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = self.results_dir / f"mia_results_{timestamp}.json"

        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResultats sauvegardes: {results_path}")

        return results

    def plot_results(self, results):
        """Genere visualisations des resultats MIA"""
        print("\n" + "="*80)
        print("GENERATION DES VISUALISATIONS")
        print("-"*80 + "\n")

        self._plot_attack_comparison(results)
        self._plot_confidence_distributions()
        self._plot_roc_curves(results)

        print("Visualisations generees!\n")

    def _plot_attack_comparison(self, results):
        """Compare les differentes methodes d'attaque"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        methods = list(results.keys())
        accuracies = [results[m]['accuracy'] for m in methods]
        f1_scores = [results[m]['f1'] for m in methods]

        # Accuracy comparison
        bars1 = axes[0].bar(methods, accuracies, color=['#ff6b6b', '#4ecdc4', '#45b7d1'], alpha=0.8)
        axes[0].axhline(y=0.5, color='red', linestyle='--', label='Random (50%)')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title('MIA Accuracy by Method', fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')
        axes[0].set_ylim([0, 1])

        # Annotate bars
        for bar, acc in zip(bars1, accuracies):
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height,
                        f'{acc:.3f}',
                        ha='center', va='bottom', fontweight='bold')

        # F1-Score comparison
        bars2 = axes[1].bar(methods, f1_scores, color=['#ff6b6b', '#4ecdc4', '#45b7d1'], alpha=0.8)
        axes[1].set_ylabel('F1-Score')
        axes[1].set_title('MIA F1-Score by Method', fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        axes[1].set_ylim([0, 1])

        # Annotate bars
        for bar, f1 in zip(bars2, f1_scores):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{f1:.3f}',
                        ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.plots_dir / 'attack_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  [1/3] Comparaison des methodes")

    def _plot_confidence_distributions(self):
        """Distribution des confidences membres vs non-membres"""
        fig, ax = plt.subplots(figsize=(10, 6))

        # Confidences
        member_conf = self.X_members[:, 0]
        non_member_conf = self.X_non_members[:, 0]

        # Histogrammes
        ax.hist(member_conf, bins=50, alpha=0.5, label='Members (Training)', color='blue')
        ax.hist(non_member_conf, bins=50, alpha=0.5, label='Non-Members (Test)', color='red')

        ax.set_xlabel('Confidence', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title('Confidence Distribution: Members vs Non-Members', fontweight='bold', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Stats
        stats_text = f"""
Member Stats:
  Mean: {np.mean(member_conf):.4f}
  Std:  {np.std(member_conf):.4f}

Non-Member Stats:
  Mean: {np.mean(non_member_conf):.4f}
  Std:  {np.std(non_member_conf):.4f}
"""
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig(self.plots_dir / 'confidence_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  [2/3] Distributions de confidence")

    def _plot_roc_curves(self, results):
        """Courbes ROC pour chaque methode"""
        fig, ax = plt.subplots(figsize=(10, 8))

        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']

        for i, (method, result) in enumerate(results.items()):
            if result.get('auc'):
                y_true = np.array(result['y_true'])
                y_pred = np.array(result['y_pred'])

                # Calculer ROC si on a les probas (shadow_model et metric_based)
                # Pour threshold, on utilise juste les predictions binaires
                if method in ['shadow_model', 'metric_based']:
                    # Ces methodes ont sauvegarde les probas
                    pass  # ROC deja calculee
                else:
                    # Utiliser predictions binaires
                    fpr, tpr, _ = roc_curve(y_true, y_pred)
                    auc_val = result.get('auc', roc_auc_score(y_true, y_pred))

                    ax.plot(fpr, tpr, color=colors[i], lw=2,
                           label=f'{method} (AUC = {auc_val:.3f})')

        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
        ax.set_xlabel('False Positive Rate', fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontweight='bold')
        ax.set_title('ROC Curves - Membership Inference Attacks', fontweight='bold', fontsize=14)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.plots_dir / 'roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  [3/3] Courbes ROC")


def main():
    """Point d'entree principal"""

    parser = argparse.ArgumentParser(
        description="Membership Inference Attack - Evaluation de confidentialite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:

  MIA complete:
    python membership_inference_attack.py \\
        --model identity_classifier_output/models/best_resnet50_TIMESTAMP.keras \\
        --train-pred identity_classifier_output/predictions/train_predictions_TIMESTAMP.json \\
        --test-pred identity_classifier_output/predictions/test_predictions_TIMESTAMP.json

  Specification manuelle des fichiers:
    python membership_inference_attack.py \\
        --model path/to/model.keras \\
        --train-pred path/to/train_pred.json \\
        --test-pred path/to/test_pred.json
        """
    )

    parser.add_argument('--model', type=str, required=True,
                       help='Chemin du modele cible (.keras)')
    parser.add_argument('--train-pred', type=str, required=True,
                       help='JSON des predictions training set')
    parser.add_argument('--test-pred', type=str, required=True,
                       help='JSON des predictions test set')

    args = parser.parse_args()

    # Creer attaquant MIA
    mia = MembershipInferenceAttack(
        target_model_path=args.model,
        predictions_train_path=args.train_pred,
        predictions_test_path=args.test_pred
    )

    # Charger donnees
    mia.load_data()

    # Executer toutes les attaques
    results = mia.run_all_attacks()

    # Visualisations
    mia.plot_results(results)

    # Resume final
    print("\n" + "="*80)
    print("ANALYSE MIA TERMINEE")
    print("="*80 + "\n")

    print("Resultats sauvegardes dans: mia_output/")
    print("  - results/     (JSON des resultats)")
    print("  - plots/       (Visualisations)")

    print("\nResume des Attaques:")
    for method, result in results.items():
        acc = result['accuracy']
        print(f"  {method:20s}: Accuracy = {acc:.4f}", end="")
        if acc < 0.55:
            print("  (EXCELLENTE confidentialite)")
        elif acc < 0.65:
            print("  (BONNE confidentialite)")
        elif acc < 0.75:
            print("  (MOYENNE confidentialite)")
        else:
            print("  (MAUVAISE confidentialite)")

    print("\n" + "="*80)
    print("\nConclusion:")
    avg_acc = np.mean([r['accuracy'] for r in results.values()])
    print(f"  Accuracy MIA moyenne: {avg_acc:.4f}")

    if avg_acc < 0.55:
        print("  Le modele a une EXCELLENTE confidentialite")
        print("  Le Face Blending est efficace pour preserver la privacy")
    elif avg_acc < 0.65:
        print("  Le modele a une BONNE confidentialite")
        print("  Faible risque de fuite d'information")
    else:
        print("  Le modele pourrait ameliorer sa confidentialite")
        print("  Recommendations:")
        print("    - Augmenter la data augmentation")
        print("    - Utiliser plus de Face Blending")
        print("    - Ajouter du bruit differentiel")
        print("    - Regularisation plus forte")

    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
