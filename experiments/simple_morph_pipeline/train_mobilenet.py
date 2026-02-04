"""Train a MobileNet classifier on the synthetic big_dataset_alpha.

Assumptions
-----------
- The dataset root (by default ../big_dataset_alpha) contains many PNG
  files named like:
    ffhq_0000__ffhq_0003__a040_orig.png
    ffhq_0000__ffhq_0003__a040_aug001.png
    ffhq_0000__ffhq_0007__a070_aug016.png
  etc.
- All images that share the same prefix BEFORE the last "_" belong to
  the same identity class, for example:
    stem = "ffhq_0000__ffhq_0003__a040_aug011"
    identity_id = "ffhq_0000__ffhq_0003__a040"

Usage (from simple_morph folder)
--------------------------------

  python train_mobilenet.py \
      --data_root ../big_dataset_alpha \
      --epochs 10 \
      --batch_size 32

This will:
  - parse the dataset and build (image_path, identity_label) pairs
  - split them into train/val sets
  - fine-tune a MobileNetV2 (pré-entraîné ImageNet) sur CPU
  - sauvegarder le meilleur modèle dans mobilenet_big_dataset_alpha.pth
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from PIL import Image


@dataclass
class Sample:
    path: Path
    label: int


class BigDatasetAlpha(Dataset):
    """Dataset pour big_dataset_alpha à partir d'un dossier plat.

    Chaque fichier .png dans data_root est pris comme un exemple.
    L'identité est déduite du nom de fichier en enlevant le dernier
    segment après "_" (orig ou augXXX).
    """

    def __init__(self, root: Path, transform=None) -> None:
        self.root = root
        self.transform = transform

        paths = sorted(p for p in root.glob("*.png"))
        if not paths:
            raise RuntimeError(f"No .png files found in {root}")

        # Construire mapping identity -> label int
        identity_to_idx: Dict[str, int] = {}
        samples: List[Sample] = []

        for p in paths:
            stem = p.stem  # ex: ffhq_0000__ffhq_0003__a040_aug011
            # identité = tout sauf le dernier segment "_xxx"
            if "_" not in stem:
                identity_id = stem
            else:
                identity_id = stem.rsplit("_", 1)[0]

            if identity_id not in identity_to_idx:
                identity_to_idx[identity_id] = len(identity_to_idx)

            label = identity_to_idx[identity_id]
            samples.append(Sample(path=p, label=label))

        self.samples = samples
        self.identity_to_idx = identity_to_idx
        self.idx_to_identity = {v: k for k, v in identity_to_idx.items()}

        print(f"Found {len(self.samples)} images, {len(self.identity_to_idx)} identities")

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.samples)

    def __getitem__(self, idx: int):  # type: ignore[override]
        sample = self.samples[idx]
        img = Image.open(sample.path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, sample.label


def create_dataloaders(
    data_root: Path,
    batch_size: int = 32,
    val_split: float = 0.2,
    num_workers: int = 4,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, int, np.ndarray, np.ndarray]:
    """Crée DataLoader train/val avec split 80/20 par identité.

    Le split est fait *par classe* (identité) : pour chaque label, on prend
    environ (1 - val_split) des images pour l'entraînement et le reste pour
    la validation. Les indices de train/val sont renvoyés pour les attaques
    MIA (membership inference).
    """

    # Transfos inspirées d'ImageNet / MobileNetV2
    train_tf = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    val_tf = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    full_dataset = BigDatasetAlpha(data_root, transform=None)
    num_classes = len(full_dataset.identity_to_idx)

    # Split indices de manière déterministe (reproductible) *par identité*.
    n_total = len(full_dataset)
    rng = np.random.default_rng(seed)

    # Construire mapping label -> indices dans le dataset complet.
    label_to_indices: Dict[int, list[int]] = {}
    for idx, sample in enumerate(full_dataset.samples):
        label_to_indices.setdefault(sample.label, []).append(idx)

    train_indices_list: list[int] = []
    val_indices_list: list[int] = []

    for label, idxs in label_to_indices.items():
        idxs_arr = np.array(idxs, dtype=int)
        rng.shuffle(idxs_arr)
        n = len(idxs_arr)
        # Nombre d'images de validation pour cette identité (≈ val_split),
        # au moins 1 si possible.
        n_val_label = max(1, int(round(n * val_split))) if n > 1 else 0
        n_train_label = n - n_val_label

        train_indices_list.extend(idxs_arr[:n_train_label].tolist())
        val_indices_list.extend(idxs_arr[n_train_label:].tolist())

    train_indices = np.array(train_indices_list, dtype=int)
    val_indices = np.array(val_indices_list, dtype=int)

    train_ds = Subset(full_dataset, train_indices.tolist())
    val_ds = Subset(full_dataset, val_indices.tolist())

    # Attacher les bons transforms
    train_ds.dataset.transform = train_tf
    val_ds.dataset.transform = val_tf

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    return train_loader, val_loader, num_classes, train_indices, val_indices


def build_mobilenet(num_classes: int) -> nn.Module:
    """Crée un MobileNetV2 pré-entraîné et adapte la dernière couche."""

    weights = MobileNet_V2_Weights.IMAGENET1K_V1
    model = mobilenet_v2(weights=weights)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        total += labels.size(0)
        correct += preds.eq(labels).sum().item()

    avg_loss = running_loss / total
    acc = correct / total
    return avg_loss, acc


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        total += labels.size(0)
        correct += preds.eq(labels).sum().item()

    avg_loss = running_loss / total
    acc = correct / total
    return avg_loss, acc


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MobileNet on big_dataset_alpha")
    parser.add_argument(
        "--data_root",
        type=str,
        default="../big_dataset_alpha",
        help="Chemin vers le dossier contenant les images fictives",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    if not data_root.exists():
        raise SystemExit(f"data_root {data_root} does not exist")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialiser les seeds pour la reproductibilité (utile pour MIA)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_loader, val_loader, num_classes, train_indices, val_indices = create_dataloaders(
        data_root,
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    print(f"Number of classes (identities): {num_classes}")

    model = build_mobilenet(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    best_val_acc = 0.0
    best_path = Path("mobilenet_big_dataset_alpha.pth")

    # Sauvegarder aussi le split train/val pour les attaques MIA
    split_path = Path(f"membership_split_{data_root.name}.npz")
    np.savez(
        split_path,
        train_indices=train_indices,
        val_indices=val_indices,
        num_classes=num_classes,
        data_root=str(data_root),
    )
    print(f"Saved train/val split to {split_path}")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device
        )

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"model_state": model.state_dict(),
                        "num_classes": num_classes}, best_path)
            print(f"  -> New best val_acc={val_acc:.4f}, model saved to {best_path}")

    print(f"Training finished. Best val_acc={best_val_acc:.4f}")


if __name__ == "__main__":
    main()
