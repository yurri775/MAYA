# -*- coding: utf-8 -*-
"""
Script d'Entrainement MobileNet pour Detection de Morphing
===========================================================

Entraine un modele MobileNet pour classifier:
- Images morphees vs Images bona fide

Architecture:
- MobileNetV2 pre-entraine sur ImageNet
- Fine-tuning sur dataset de morphing
- Data augmentation pour equilibrer les classes
- Metriques completes et visualisations

Version professionnelle sans emojis
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
import cv2
from pathlib import Path
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# TensorFlow et Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, f1_score, accuracy_score
)
from datetime import datetime


class MorphingDetectorTrainer:
    """
    Classe pour entrainer un detecteur de morphing base sur MobileNetV2
    """

    def __init__(self, img_size=(224, 224), batch_size=32, epochs=50):
        """
        Initialisation du trainer

        Args:
            img_size: Taille des images (height, width)
            batch_size: Taille des batches
            epochs: Nombre d'epoques d'entrainement
        """
        self.img_size = img_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = None
        self.history = None

        # Creer dossiers de sortie
        self.output_dir = Path("model_output")
        self.output_dir.mkdir(exist_ok=True)

        self.models_dir = self.output_dir / "models"
        self.models_dir.mkdir(exist_ok=True)

        self.plots_dir = self.output_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)

        self.logs_dir = self.output_dir / "logs"
        self.logs_dir.mkdir(exist_ok=True)

        print("\n" + "="*80)
        print("DETECTION DE MORPHING - ENTRAINEMENT MOBILENET")
        print("="*80 + "\n")

    def load_dataset(self, morph_dir, bona_fide_dir, test_size=0.2, val_size=0.1):
        """
        Charge et prepare le dataset

        Args:
            morph_dir: Dossier des images morphees
            bona_fide_dir: Dossier des images bona fide
            test_size: Proportion du test set
            val_size: Proportion du validation set

        Returns:
            Tuple (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        print("CHARGEMENT DU DATASET")
        print("-"*80 + "\n")

        # Lister les images
        morph_paths = list(Path(morph_dir).glob("*.png")) + list(Path(morph_dir).glob("*.jpg"))
        bona_fide_paths = list(Path(bona_fide_dir).glob("*.png")) + list(Path(bona_fide_dir).glob("*.jpg"))

        print(f"Images morphees trouvees:  {len(morph_paths)}")
        print(f"Images bona fide trouvees: {len(bona_fide_paths)}")

        if len(morph_paths) < 10 or len(bona_fide_paths) < 5:
            raise ValueError("Pas assez d'images pour l'entrainement (min 10 morphs, 5 bona fide)")

        # Charger les images
        print("\nChargement des images...")
        X = []
        y = []

        # Charger morphs (label = 1)
        for img_path in tqdm(morph_paths, desc="Morphs"):
            img = cv2.imread(str(img_path))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, self.img_size)
                X.append(img)
                y.append(1)  # Morph = 1

        # Charger bona fide (label = 0)
        for img_path in tqdm(bona_fide_paths, desc="Bona fide"):
            img = cv2.imread(str(img_path))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, self.img_size)
                X.append(img)
                y.append(0)  # Bona fide = 0

        X = np.array(X, dtype=np.float32) / 255.0  # Normalisation [0, 1]
        y = np.array(y)

        print(f"\nDataset charge: {len(X)} images")
        print(f"  Morphs:     {np.sum(y == 1)}")
        print(f"  Bona fide:  {np.sum(y == 0)}")
        print(f"  Ratio:      {np.sum(y == 1) / np.sum(y == 0):.2f}")

        # Split train/temp
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=test_size + val_size, random_state=42, stratify=y
        )

        # Split temp en val/test
        relative_val_size = val_size / (test_size + val_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=1-relative_val_size, random_state=42, stratify=y_temp
        )

        print(f"\nSplits:")
        print(f"  Train:      {len(X_train)} images ({np.sum(y_train == 1)} morphs, {np.sum(y_train == 0)} bona fide)")
        print(f"  Validation: {len(X_val)} images ({np.sum(y_val == 1)} morphs, {np.sum(y_val == 0)} bona fide)")
        print(f"  Test:       {len(X_test)} images ({np.sum(y_test == 1)} morphs, {np.sum(y_test == 0)} bona fide)")

        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test

        return X_train, X_val, X_test, y_train, y_val, y_test

    def create_data_generators(self):
        """
        Cree des generateurs avec data augmentation
        """
        print("\n" + "="*80)
        print("DATA AUGMENTATION")
        print("-"*80 + "\n")

        # Data augmentation pour training
        train_datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            shear_range=0.2,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest'
        )

        # Pas d'augmentation pour validation/test
        val_test_datagen = ImageDataGenerator()

        print("Augmentations appliquees:")
        print("  - Rotation: ±20 degres")
        print("  - Translation: ±20%")
        print("  - Flip horizontal")
        print("  - Zoom: ±20%")
        print("  - Shear: ±20%")
        print("  - Brightness: ±20%")

        self.train_generator = train_datagen.flow(
            self.X_train, self.y_train, batch_size=self.batch_size
        )
        self.val_generator = val_test_datagen.flow(
            self.X_val, self.y_val, batch_size=self.batch_size
        )

        return self.train_generator, self.val_generator

    def build_model(self, trainable_layers=20):
        """
        Construit le modele MobileNetV2 avec fine-tuning

        Args:
            trainable_layers: Nombre de couches a entrainer (depuis la fin)
        """
        print("\n" + "="*80)
        print("CONSTRUCTION DU MODELE")
        print("-"*80 + "\n")

        # Charger MobileNetV2 pre-entraine
        print("Chargement de MobileNetV2 pre-entraine sur ImageNet...")
        base_model = MobileNetV2(
            input_shape=(*self.img_size, 3),
            include_top=False,
            weights='imagenet'
        )

        # Geler les couches de base
        base_model.trainable = True
        for layer in base_model.layers[:-trainable_layers]:
            layer.trainable = False

        print(f"  Couches totales:      {len(base_model.layers)}")
        print(f"  Couches entrainables: {trainable_layers}")

        # Construire le modele complet
        inputs = keras.Input(shape=(*self.img_size, 3))
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)

        model = models.Model(inputs, outputs)

        # Compiler
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-4),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )

        print("\nArchitecture du modele:")
        print("  MobileNetV2 (pre-entraine)")
        print("  GlobalAveragePooling2D")
        print("  Dense(256) + Dropout(0.5)")
        print("  Dense(128) + Dropout(0.3)")
        print("  Dense(1, sigmoid)")

        print(f"\nParametres entrainables: {model.count_params():,}")

        self.model = model
        return model

    def train(self):
        """
        Entraine le modele
        """
        print("\n" + "="*80)
        print("ENTRAINEMENT DU MODELE")
        print("-"*80 + "\n")

        # Callbacks
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        checkpoint = ModelCheckpoint(
            filepath=str(self.models_dir / f"best_model_{timestamp}.keras"),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )

        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )

        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )

        tensorboard = TensorBoard(
            log_dir=str(self.logs_dir / f"logs_{timestamp}"),
            histogram_freq=1
        )

        callbacks = [checkpoint, early_stop, reduce_lr, tensorboard]

        # Calculer steps
        steps_per_epoch = len(self.X_train) // self.batch_size
        validation_steps = len(self.X_val) // self.batch_size

        print(f"Configuration:")
        print(f"  Epochs:            {self.epochs}")
        print(f"  Batch size:        {self.batch_size}")
        print(f"  Steps per epoch:   {steps_per_epoch}")
        print(f"  Validation steps:  {validation_steps}")
        print(f"  Learning rate:     1e-4")
        print("\nCallbacks:")
        print("  - ModelCheckpoint (save best)")
        print("  - EarlyStopping (patience=10)")
        print("  - ReduceLROnPlateau (patience=5)")
        print("  - TensorBoard")
        print("\nDemarrage de l'entrainement...\n")

        # Entrainer
        history = self.model.fit(
            self.train_generator,
            epochs=self.epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=self.val_generator,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1
        )

        self.history = history
        print("\nEntrainement termine!")

        # Sauvegarder le modele final
        final_model_path = self.models_dir / f"final_model_{timestamp}.keras"
        self.model.save(final_model_path)
        print(f"\nModele final sauvegarde: {final_model_path}")

        return history

    def evaluate(self):
        """
        Evalue le modele sur le test set
        """
        print("\n" + "="*80)
        print("EVALUATION DU MODELE")
        print("-"*80 + "\n")

        # Predictions
        y_pred_proba = self.model.predict(self.X_test, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()

        # Metriques
        accuracy = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)

        print("Resultats sur Test Set:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  F1-Score:  {f1:.4f}")

        print("\nClassification Report:")
        print(classification_report(
            self.y_test, y_pred,
            target_names=['Bona Fide', 'Morph'],
            digits=4
        ))

        # Matrice de confusion
        cm = confusion_matrix(self.y_test, y_pred)
        print("Matrice de Confusion:")
        print(f"                  Predicted")
        print(f"                Bona Fide  Morph")
        print(f"Actual Bona Fide    {cm[0, 0]:4d}     {cm[0, 1]:4d}")
        print(f"       Morph        {cm[1, 0]:4d}     {cm[1, 1]:4d}")

        return {
            'accuracy': accuracy,
            'f1': f1,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'confusion_matrix': cm
        }

    def plot_results(self, eval_results):
        """
        Genere toutes les visualisations
        """
        print("\n" + "="*80)
        print("GENERATION DES VISUALISATIONS")
        print("-"*80 + "\n")

        # 1. Courbes d'entrainement
        self._plot_training_curves()

        # 2. Matrice de confusion
        self._plot_confusion_matrix(eval_results['confusion_matrix'])

        # 3. Courbe ROC
        self._plot_roc_curve(eval_results['y_pred_proba'])

        # 4. Precision-Recall curve
        self._plot_precision_recall_curve(eval_results['y_pred_proba'])

        print("Toutes les visualisations generees!\n")

    def _plot_training_curves(self):
        """Courbes d'entrainement"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Train')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation')
        axes[0, 0].set_title('Model Accuracy', fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Train')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation')
        axes[0, 1].set_title('Model Loss', fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Precision
        axes[1, 0].plot(self.history.history['precision'], label='Train')
        axes[1, 0].plot(self.history.history['val_precision'], label='Validation')
        axes[1, 0].set_title('Model Precision', fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Recall
        axes[1, 1].plot(self.history.history['recall'], label='Train')
        axes[1, 1].plot(self.history.history['val_recall'], label='Validation')
        axes[1, 1].set_title('Model Recall', fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.plots_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  [1/4] Courbes d'entrainement sauvegardees")

    def _plot_confusion_matrix(self, cm):
        """Matrice de confusion"""
        fig, ax = plt.subplots(figsize=(8, 6))

        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Bona Fide', 'Morph'],
            yticklabels=['Bona Fide', 'Morph'],
            ax=ax, cbar_kws={'label': 'Count'}
        )

        ax.set_title('Confusion Matrix', fontweight='bold', fontsize=14)
        ax.set_xlabel('Predicted', fontweight='bold')
        ax.set_ylabel('Actual', fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.plots_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  [2/4] Matrice de confusion sauvegardee")

    def _plot_roc_curve(self, y_pred_proba):
        """Courbe ROC"""
        fpr, tpr, thresholds = roc_curve(self.y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(8, 6))

        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontweight='bold')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontweight='bold', fontsize=14)
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.plots_dir / 'roc_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  [3/4] Courbe ROC sauvegardee")

    def _plot_precision_recall_curve(self, y_pred_proba):
        """Courbe Precision-Recall"""
        precision, recall, thresholds = precision_recall_curve(self.y_test, y_pred_proba)

        fig, ax = plt.subplots(figsize=(8, 6))

        ax.plot(recall, precision, color='blue', lw=2)
        ax.set_xlabel('Recall', fontweight='bold')
        ax.set_ylabel('Precision', fontweight='bold')
        ax.set_title('Precision-Recall Curve', fontweight='bold', fontsize=14)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.grid(True, alpha=0.3)

        # F1 optimal
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        best_idx = np.argmax(f1_scores)
        best_f1 = f1_scores[best_idx]
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

        ax.plot(recall[best_idx], precision[best_idx], 'ro', markersize=10,
               label=f'Best F1={best_f1:.4f} (threshold={best_threshold:.4f})')
        ax.legend()

        plt.tight_layout()
        plt.savefig(self.plots_dir / 'precision_recall_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  [4/4] Courbe Precision-Recall sauvegardee")


def main():
    """Point d'entree principal"""

    parser = argparse.ArgumentParser(
        description="Entrainement MobileNet pour detection de morphing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:

  Entrainement de base:
    python train_mobilenet_detector.py --morph morphing_results --bona-fide sample_data/before_morph

  Avec parametres personnalises:
    python train_mobilenet_detector.py --morph morphing_results --bona-fide sample_data/before_morph --epochs 100 --batch-size 16

  Image size personnalisee:
    python train_mobilenet_detector.py --morph morphing_results --bona-fide sample_data/before_morph --img-size 160
        """
    )

    parser.add_argument('--morph', type=str, required=True,
                       help='Dossier des images morphees')
    parser.add_argument('--bona-fide', type=str, required=True,
                       help='Dossier des images bona fide')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Nombre d\'epoques (defaut: 50)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Taille des batches (defaut: 32)')
    parser.add_argument('--img-size', type=int, default=224,
                       help='Taille des images (defaut: 224)')
    parser.add_argument('--trainable-layers', type=int, default=20,
                       help='Nombre de couches MobileNet a entrainer (defaut: 20)')

    args = parser.parse_args()

    # Creer trainer
    trainer = MorphingDetectorTrainer(
        img_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        epochs=args.epochs
    )

    # Pipeline complet
    trainer.load_dataset(args.morph, args.bona_fide)
    trainer.create_data_generators()
    trainer.build_model(trainable_layers=args.trainable_layers)
    trainer.train()
    eval_results = trainer.evaluate()
    trainer.plot_results(eval_results)

    print("="*80)
    print("ENTRAINEMENT TERMINE AVEC SUCCES!")
    print("="*80 + "\n")
    print("Resultats sauvegardes dans: model_output/")
    print("  - models/                   (Modeles sauvegardes)")
    print("  - plots/                    (Visualisations)")
    print("  - logs/                     (TensorBoard logs)")
    print("\nPour visualiser avec TensorBoard:")
    print("  tensorboard --logdir=model_output/logs")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
