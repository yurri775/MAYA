# -*- coding: utf-8 -*-
"""
Entrainement CNN pour Classification d'Identites Fictives
==========================================================

Ce module entraine un reseau de neurones convolutif profond (CNN) pour classifier
des identites fictives generees par face blending/morphing.

Objectif: Evaluer la capacite du modele a memoriser les identites sources
pour ensuite mesurer la vulnerabilite avec une Membership Inference Attack (MIA).

Base sur:
- Ghorbel et al. (2024) - Face Blending Data Augmentation
- Shokri et al. (2017) - Membership Inference Attacks

Architecture supportees:
- ResNet50
- InceptionV3
- VGG16
- MobileNetV2
- EfficientNetB0

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
import cv2
from pathlib import Path
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime

# TensorFlow et Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import (
    ResNet50, InceptionV3, VGG16, MobileNetV2, EfficientNetB0
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,
    TensorBoard, CSVLogger
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, top_k_accuracy_score
from sklearn.preprocessing import LabelEncoder


class IdentityClassifierTrainer:
    """
    Classe pour entrainer un CNN pour classification d'identites fictives
    avec suivi des metriques pour evaluation MIA ulterieure
    """

    def __init__(self, architecture='resnet50', img_size=(224, 224),
                 batch_size=32, epochs=100, include_blended=True):
        """
        Initialisation du trainer

        Args:
            architecture: Architecture CNN ('resnet50', 'inceptionv3', 'vgg16', 'mobilenetv2', 'efficientnetb0')
            img_size: Taille des images (height, width)
            batch_size: Taille des batches
            epochs: Nombre d'epoques
            include_blended: Inclure images blendees/morphees dans training
        """
        self.architecture = architecture
        self.img_size = img_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.include_blended = include_blended
        self.model = None
        self.history = None
        self.label_encoder = LabelEncoder()

        # Creer dossiers de sortie
        self.output_dir = Path("identity_classifier_output")
        self.output_dir.mkdir(exist_ok=True)

        self.models_dir = self.output_dir / "models"
        self.models_dir.mkdir(exist_ok=True)

        self.plots_dir = self.output_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)

        self.logs_dir = self.output_dir / "logs"
        self.logs_dir.mkdir(exist_ok=True)

        self.predictions_dir = self.output_dir / "predictions"
        self.predictions_dir.mkdir(exist_ok=True)

        print("\n" + "="*80)
        print("CLASSIFICATION D'IDENTITES FICTIVES - ENTRAINEMENT CNN")
        print("Projet de Recherche: Face Blending + MIA")
        print("="*80 + "\n")
        print(f"Architecture:      {architecture.upper()}")
        print(f"Image size:        {img_size}")
        print(f"Batch size:        {batch_size}")
        print(f"Epochs:            {epochs}")
        print(f"Include blended:   {include_blended}")

    def load_identity_dataset(self, data_dir, test_size=0.2, val_size=0.1):
        """
        Charge le dataset d'identites organisé par dossiers

        Structure attendue:
        data_dir/
            identity_001/
                img_001.png
                img_002.png
            identity_002/
                ...

        Args:
            data_dir: Dossier racine contenant les sous-dossiers d'identites
            test_size: Proportion du test set
            val_size: Proportion du validation set

        Returns:
            Tuple (X_train, X_val, X_test, y_train, y_val, y_test, identity_names)
        """
        print("\n" + "="*80)
        print("CHARGEMENT DU DATASET D'IDENTITES")
        print("-"*80 + "\n")

        data_path = Path(data_dir)
        if not data_path.exists():
            raise ValueError(f"Le dossier n'existe pas: {data_dir}")

        # Lister les identites (sous-dossiers)
        identity_folders = [d for d in data_path.iterdir() if d.is_dir()]
        identity_folders = sorted(identity_folders, key=lambda x: x.name)

        if len(identity_folders) < 2:
            raise ValueError(f"Au moins 2 identites requises, trouve: {len(identity_folders)}")

        print(f"Identites trouvees: {len(identity_folders)}")
        print(f"Exemples: {', '.join([f.name for f in identity_folders[:5]])}")

        # Charger les images
        X = []
        y = []
        identity_names = []
        images_per_identity = {}

        for identity_idx, identity_folder in enumerate(tqdm(identity_folders, desc="Chargement")):
            identity_name = identity_folder.name
            identity_names.append(identity_name)

            # Trouver toutes les images
            image_paths = []
            for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
                image_paths.extend(list(identity_folder.glob(ext)))

            if len(image_paths) == 0:
                print(f"  Attention: Aucune image pour {identity_name}")
                continue

            images_per_identity[identity_name] = len(image_paths)

            # Charger images
            for img_path in image_paths:
                img = cv2.imread(str(img_path))
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, self.img_size)
                    X.append(img)
                    y.append(identity_idx)

        X = np.array(X, dtype=np.float32) / 255.0
        y = np.array(y)

        print(f"\nDataset charge:")
        print(f"  Total images:   {len(X)}")
        print(f"  Identites:      {len(identity_names)}")
        print(f"  Images/ID:      Min={min(images_per_identity.values())}, "
              f"Max={max(images_per_identity.values())}, "
              f"Moyenne={np.mean(list(images_per_identity.values())):.1f}")

        # Verifier equilibre
        unique, counts = np.unique(y, return_counts=True)
        print(f"\nDistribution des classes:")
        for id_idx, count in zip(unique[:5], counts[:5]):
            print(f"  {identity_names[id_idx]}: {count} images")
        if len(unique) > 5:
            print(f"  ... ({len(unique)-5} autres identites)")

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
        print(f"  Train:      {len(X_train)} images")
        print(f"  Validation: {len(X_val)} images")
        print(f"  Test:       {len(X_test)} images")

        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        self.identity_names = identity_names
        self.n_classes = len(identity_names)

        # Sauvegarder metadata
        metadata = {
            'n_identities': len(identity_names),
            'identity_names': identity_names,
            'images_per_identity': images_per_identity,
            'train_size': len(X_train),
            'val_size': len(X_val),
            'test_size': len(X_test),
            'img_size': self.img_size,
            'architecture': self.architecture
        }

        with open(self.output_dir / 'dataset_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        return X_train, X_val, X_test, y_train, y_val, y_test, identity_names

    def create_data_generators(self):
        """
        Cree des generateurs avec data augmentation
        Compatible avec Face Blending data augmentation
        """
        print("\n" + "="*80)
        print("DATA AUGMENTATION (Face Blending Compatible)")
        print("-"*80 + "\n")

        # Data augmentation - similaire a Ghorbel et al. (2024)
        train_datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.15,
            height_shift_range=0.15,
            horizontal_flip=True,
            zoom_range=0.15,
            shear_range=0.1,
            brightness_range=[0.85, 1.15],
            fill_mode='nearest'
        )

        val_test_datagen = ImageDataGenerator()

        print("Augmentations appliquees (compatibles Face Blending):")
        print("  - Rotation: ±15 degres")
        print("  - Translation: ±15%")
        print("  - Flip horizontal")
        print("  - Zoom: ±15%")
        print("  - Shear: ±10%")
        print("  - Brightness: ±15%")

        self.train_generator = train_datagen.flow(
            self.X_train, self.y_train, batch_size=self.batch_size
        )
        self.val_generator = val_test_datagen.flow(
            self.X_val, self.y_val, batch_size=self.batch_size
        )

        return self.train_generator, self.val_generator

    def build_model(self, trainable_layers=30):
        """
        Construit le modele CNN avec architecture specifiee

        Args:
            trainable_layers: Nombre de couches a fine-tuner
        """
        print("\n" + "="*80)
        print(f"CONSTRUCTION DU MODELE - {self.architecture.upper()}")
        print("-"*80 + "\n")

        # Selectionner l'architecture
        architectures = {
            'resnet50': ResNet50,
            'inceptionv3': InceptionV3,
            'vgg16': VGG16,
            'mobilenetv2': MobileNetV2,
            'efficientnetb0': EfficientNetB0
        }

        if self.architecture not in architectures:
            raise ValueError(f"Architecture non supportee: {self.architecture}")

        base_model_class = architectures[self.architecture]

        # Ajuster img_size pour InceptionV3 (min 75x75)
        if self.architecture == 'inceptionv3' and min(self.img_size) < 75:
            print(f"Attention: InceptionV3 requiert min 75x75, ajuste a 75x75")
            self.img_size = (75, 75)

        print(f"Chargement de {self.architecture.upper()} pre-entraine sur ImageNet...")

        base_model = base_model_class(
            input_shape=(*self.img_size, 3),
            include_top=False,
            weights='imagenet'
        )

        # Fine-tuning
        base_model.trainable = True
        for layer in base_model.layers[:-trainable_layers]:
            layer.trainable = False

        print(f"  Couches totales:      {len(base_model.layers)}")
        print(f"  Couches entrainables: {trainable_layers}")

        # Construire le modele
        inputs = keras.Input(shape=(*self.img_size, 3))
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(self.n_classes, activation='softmax')(x)

        model = models.Model(inputs, outputs)

        # Compiler
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-4),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=5, name='top5_accuracy')]
        )

        print("\nArchitecture complete:")
        print(f"  {self.architecture.upper()} (pre-entraine)")
        print("  GlobalAveragePooling2D")
        print("  Dense(512) + Dropout(0.5)")
        print("  Dense(256) + Dropout(0.3)")
        print(f"  Dense({self.n_classes}, softmax)")

        print(f"\nParametres entrainables: {model.count_params():,}")

        self.model = model
        return model

    def train(self):
        """
        Entraine le modele avec suivi detaille pour MIA
        """
        print("\n" + "="*80)
        print("ENTRAINEMENT DU MODELE")
        print("-"*80 + "\n")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Callbacks
        checkpoint = ModelCheckpoint(
            filepath=str(self.models_dir / f"best_{self.architecture}_{timestamp}.keras"),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )

        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )

        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-7,
            verbose=1
        )

        tensorboard = TensorBoard(
            log_dir=str(self.logs_dir / f"logs_{timestamp}"),
            histogram_freq=1,
            write_graph=True
        )

        csv_logger = CSVLogger(
            str(self.logs_dir / f"training_log_{timestamp}.csv")
        )

        callbacks = [checkpoint, early_stop, reduce_lr, tensorboard, csv_logger]

        # Steps
        steps_per_epoch = len(self.X_train) // self.batch_size
        validation_steps = len(self.X_val) // self.batch_size

        print(f"Configuration:")
        print(f"  Architecture:      {self.architecture}")
        print(f"  Identites:         {self.n_classes}")
        print(f"  Epochs:            {self.epochs}")
        print(f"  Batch size:        {self.batch_size}")
        print(f"  Steps per epoch:   {steps_per_epoch}")
        print(f"  Validation steps:  {validation_steps}")
        print(f"  Learning rate:     1e-4")

        print("\nCallbacks actifs:")
        print("  - ModelCheckpoint (save best)")
        print("  - EarlyStopping (patience=15)")
        print("  - ReduceLROnPlateau (patience=7)")
        print("  - TensorBoard")
        print("  - CSVLogger")

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
        self.timestamp = timestamp

        print("\nEntrainement termine!")

        # Sauvegarder modele final
        final_model_path = self.models_dir / f"final_{self.architecture}_{timestamp}.keras"
        self.model.save(final_model_path)
        print(f"\nModele final sauvegarde: {final_model_path}")

        return history

    def evaluate_and_save_predictions(self):
        """
        Evalue le modele et sauvegarde les predictions
        Important pour MIA: besoin des probabilites et confiances
        """
        print("\n" + "="*80)
        print("EVALUATION ET SAUVEGARDE DES PREDICTIONS")
        print("-"*80 + "\n")

        # Predictions sur test set
        print("Predictions sur test set...")
        y_pred_proba = self.model.predict(self.X_test, verbose=1)
        y_pred = np.argmax(y_pred_proba, axis=1)

        # Metriques
        from sklearn.metrics import accuracy_score, top_k_accuracy_score

        test_accuracy = accuracy_score(self.y_test, y_pred)
        test_top5 = top_k_accuracy_score(self.y_test, y_pred_proba, k=min(5, self.n_classes))

        print(f"\nResultats sur Test Set:")
        print(f"  Accuracy (Top-1):  {test_accuracy:.4f}")
        print(f"  Accuracy (Top-5):  {test_top5:.4f}")

        # Classification report
        print("\nClassification Report (premiers 10 classes):")
        print(classification_report(
            self.y_test, y_pred,
            target_names=self.identity_names[:min(10, len(self.identity_names))],
            digits=4,
            labels=list(range(min(10, len(self.identity_names))))
        ))

        # Sauvegarder predictions pour MIA
        predictions_data = {
            'y_true': self.y_test.tolist(),
            'y_pred': y_pred.tolist(),
            'y_pred_proba': y_pred_proba.tolist(),
            'confidence': np.max(y_pred_proba, axis=1).tolist(),
            'identity_names': self.identity_names
        }

        pred_path = self.predictions_dir / f"test_predictions_{self.timestamp}.json"
        with open(pred_path, 'w') as f:
            json.dump(predictions_data, f, indent=2)

        print(f"\nPredictions sauvegardees: {pred_path}")
        print("Ces predictions seront utilisees pour l'evaluation MIA")

        # Aussi sauvegarder predictions train pour MIA
        print("\nSauvegarde predictions training set (pour MIA)...")
        train_pred_proba = self.model.predict(self.X_train[:1000], verbose=0)  # Limiter pour memoire

        train_predictions = {
            'y_true': self.y_train[:1000].tolist(),
            'y_pred_proba': train_pred_proba.tolist(),
            'confidence': np.max(train_pred_proba, axis=1).tolist()
        }

        train_pred_path = self.predictions_dir / f"train_predictions_{self.timestamp}.json"
        with open(train_pred_path, 'w') as f:
            json.dump(train_predictions, f, indent=2)

        print(f"Predictions training sauvegardees: {train_pred_path}")

        return {
            'test_accuracy': test_accuracy,
            'test_top5': test_top5,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'confidence': np.max(y_pred_proba, axis=1)
        }

    def plot_results(self):
        """Genere visualisations"""
        print("\n" + "="*80)
        print("GENERATION DES VISUALISATIONS")
        print("-"*80 + "\n")

        self._plot_training_curves()
        self._plot_confusion_matrix_sample()
        self._plot_confidence_distribution()

        print("Visualisations generees!\n")

    def _plot_training_curves(self):
        """Courbes d'entrainement"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Accuracy
        axes[0].plot(self.history.history['accuracy'], label='Train')
        axes[0].plot(self.history.history['val_accuracy'], label='Validation')
        axes[0].set_title('Model Accuracy', fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Loss
        axes[1].plot(self.history.history['loss'], label='Train')
        axes[1].plot(self.history.history['val_loss'], label='Validation')
        axes[1].set_title('Model Loss', fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Top-5 Accuracy
        if 'top5_accuracy' in self.history.history:
            axes[2].plot(self.history.history['top5_accuracy'], label='Train')
            axes[2].plot(self.history.history['val_top5_accuracy'], label='Validation')
            axes[2].set_title('Top-5 Accuracy', fontweight='bold')
            axes[2].set_xlabel('Epoch')
            axes[2].set_ylabel('Top-5 Accuracy')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.plots_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  [1/3] Courbes d'entrainement")

    def _plot_confusion_matrix_sample(self):
        """Matrice de confusion (echantillon si trop de classes)"""
        y_pred_proba = self.model.predict(self.X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)

        # Si trop de classes, prendre echantillon
        n_display = min(10, self.n_classes)
        selected_classes = list(range(n_display))

        # Filter
        mask = np.isin(self.y_test, selected_classes)
        y_true_filtered = self.y_test[mask]
        y_pred_filtered = y_pred[mask]

        cm = confusion_matrix(y_true_filtered, y_pred_filtered, labels=selected_classes)

        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=[self.identity_names[i] for i in selected_classes],
                   yticklabels=[self.identity_names[i] for i in selected_classes])

        ax.set_title(f'Confusion Matrix (first {n_display} identities)', fontweight='bold')
        ax.set_xlabel('Predicted', fontweight='bold')
        ax.set_ylabel('Actual', fontweight='bold')

        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'confusion_matrix_sample.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  [2/3] Matrice de confusion")

    def _plot_confidence_distribution(self):
        """Distribution des confidences (important pour MIA)"""
        y_pred_proba = self.model.predict(self.X_test, verbose=0)
        confidences = np.max(y_pred_proba, axis=1)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Histogramme
        axes[0].hist(confidences, bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[0].set_title('Confidence Distribution', fontweight='bold')
        axes[0].set_xlabel('Confidence')
        axes[0].set_ylabel('Frequency')
        axes[0].axvline(np.mean(confidences), color='red', linestyle='--',
                       label=f'Mean: {np.mean(confidences):.3f}')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Box plot
        axes[1].boxplot(confidences, vert=True)
        axes[1].set_title('Confidence Box Plot', fontweight='bold')
        axes[1].set_ylabel('Confidence')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.plots_dir / 'confidence_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  [3/3] Distribution des confidences")


def main():
    """Point d'entree principal"""

    parser = argparse.ArgumentParser(
        description="Entrainement CNN pour classification d'identites fictives",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:

  ResNet50 (recommande):
    python train_identity_classifier.py --data identities_dataset --arch resnet50 --epochs 100

  InceptionV3:
    python train_identity_classifier.py --data identities_dataset --arch inceptionv3 --epochs 100

  Test rapide:
    python train_identity_classifier.py --data identities_dataset --epochs 20
        """
    )

    parser.add_argument('--data', type=str, required=True,
                       help='Dossier racine du dataset d\'identites')
    parser.add_argument('--arch', type=str, default='resnet50',
                       choices=['resnet50', 'inceptionv3', 'vgg16', 'mobilenetv2', 'efficientnetb0'],
                       help='Architecture CNN (defaut: resnet50)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Nombre d\'epoques (defaut: 100)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Taille des batches (defaut: 32)')
    parser.add_argument('--img-size', type=int, default=224,
                       help='Taille des images (defaut: 224)')
    parser.add_argument('--trainable-layers', type=int, default=30,
                       help='Nombre de couches a fine-tuner (defaut: 30)')
    parser.add_argument('--include-blended', action='store_true',
                       help='Inclure images blendees dans training')

    args = parser.parse_args()

    # Creer trainer
    trainer = IdentityClassifierTrainer(
        architecture=args.arch,
        img_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        epochs=args.epochs,
        include_blended=args.include_blended
    )

    # Pipeline complet
    trainer.load_identity_dataset(args.data)
    trainer.create_data_generators()
    trainer.build_model(trainable_layers=args.trainable_layers)
    trainer.train()
    trainer.evaluate_and_save_predictions()
    trainer.plot_results()

    print("="*80)
    print("ENTRAINEMENT TERMINE!")
    print("="*80 + "\n")
    print("Resultats sauvegardes dans: identity_classifier_output/")
    print("  - models/        (Modeles entraines)")
    print("  - predictions/   (Predictions pour MIA)")
    print("  - plots/         (Visualisations)")
    print("  - logs/          (TensorBoard logs)")
    print("\nProchaine etape: Membership Inference Attack (MIA)")
    print("  python membership_inference_attack.py --model identity_classifier_output/models/...")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
