"""Simple Membership Inference Attack (MIA) on MobileNet trained on synthetic dataset.

We assume:
- The model was trained with train_mobilenet.py
- A membership split file membership_split_<data_root.name>.npz exists
  (train_indices, val_indices).
- We run the attack on big_dataset_lfw by default.

Attack strategy
---------------
- Load the trained MobileNet and dataset.
- For every image x with label y, compute:
    * cross-entropy loss L(x,y)
    * confidence = max softmax probability
- Use the known membership labels (train vs val) to build two distributions:
    * losses for members (train)
    * losses for non-members (val)
- Find the threshold tau on loss that maximizes attack accuracy on this set:
    predict "member" if L(x,y) < tau.
- Report:
    * attack accuracy
    * true positive rate (members correctly identified)
    * true negative rate (non-members correctly identified)

Run from simple_morph:

    python mia_attack.py \
        --data_root big_dataset_lfw \
        --split membership_split_big_dataset_lfw.npz \
        --checkpoint mobilenet_big_dataset_alpha.pth
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

from train_mobilenet import BigDatasetAlpha, build_mobilenet


def load_model(checkpoint_path: Path, num_classes: int, device: torch.device) -> torch.nn.Module:
    model = build_mobilenet(num_classes)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def compute_losses_and_confidences(
    dataset: BigDatasetAlpha,
    model: torch.nn.Module,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute CE loss and max-softmax confidence for each sample in the dataset."""

    criterion = nn.CrossEntropyLoss(reduction="none")
    n = len(dataset)
    losses = np.zeros(n, dtype=np.float32)
    confidences = np.zeros(n, dtype=np.float32)

    with torch.no_grad():
        for idx in range(n):
            img, label = dataset[idx]
            img = img.unsqueeze(0).to(device)
            label_t = torch.tensor([label], dtype=torch.long, device=device)

            logits = model(img)
            prob = torch.softmax(logits, dim=1)

            loss = criterion(logits, label_t)  # shape (1,)
            losses[idx] = loss.item()
            confidences[idx] = prob.max(dim=1).values.item()

    return losses, confidences


def best_threshold_from_losses(
    member_losses: np.ndarray,
    nonmember_losses: np.ndarray,
) -> tuple[float, float, float, float]:
    """Find loss threshold that maximizes membership prediction accuracy.

    Returns (best_tau, best_acc, tpr, tnr).
    """

    y_true = np.concatenate([
        np.ones_like(member_losses, dtype=int),
        np.zeros_like(nonmember_losses, dtype=int),
    ])
    scores = np.concatenate([member_losses, nonmember_losses])

    # lower loss => more likely member, so we will predict member if score < tau
    order = np.argsort(scores)
    sorted_scores = scores[order]
    sorted_labels = y_true[order]

    # candidates between consecutive distinct scores
    best_acc = -1.0
    best_tau = sorted_scores[0]
    best_tpr = 0.0
    best_tnr = 0.0

    # Precompute cumulative counts of positives when threshold sweeps from low to high
    cum_pos = np.cumsum(sorted_labels == 1)
    total_pos = cum_pos[-1]
    total_neg = len(sorted_labels) - total_pos

    for i in range(len(sorted_scores)):
        tau = sorted_scores[i]
        # Predict member for scores < tau
        # number of samples with score < tau is i (indices 0..i-1)
        n_member_pred = i
        tp = cum_pos[i - 1] if i > 0 else 0
        fp = n_member_pred - tp

        fn = total_pos - tp
        tn = total_neg - fp

        acc = (tp + tn) / len(sorted_scores)
        if acc > best_acc:
            best_acc = acc
            best_tau = tau
            best_tpr = tp / total_pos if total_pos > 0 else 0.0
            best_tnr = tn / total_neg if total_neg > 0 else 0.0

    return best_tau, best_acc, best_tpr, best_tnr


def main() -> None:
    parser = argparse.ArgumentParser(description="Membership Inference Attack on MobileNet")
    parser.add_argument("--data_root", type=str, default="big_dataset_lfw")
    parser.add_argument("--split", type=str, default="membership_split_big_dataset_lfw.npz")
    parser.add_argument("--checkpoint", type=str, default="mobilenet_big_dataset_alpha.pth")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    split_path = Path(args.split)
    ckpt_path = Path(args.checkpoint)

    if not data_root.exists():
        raise SystemExit(f"data_root {data_root} does not exist")
    if not split_path.exists():
        raise SystemExit(f"split file {split_path} does not exist")
    if not ckpt_path.exists():
        raise SystemExit(f"checkpoint {ckpt_path} does not exist")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    split = np.load(split_path)
    train_indices = split["train_indices"].astype(int)
    val_indices = split["val_indices"].astype(int)
    num_classes = int(split["num_classes"])

    print(f"Loaded split: {len(train_indices)} train members, {len(val_indices)} val non-members")

    # Evaluation transform (same as validation in training script)
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

    dataset = BigDatasetAlpha(data_root, transform=val_tf)
    model = load_model(ckpt_path, num_classes, device)

    print("Computing per-sample losses and confidences...")
    losses, confidences = compute_losses_and_confidences(dataset, model, device)

    member_losses = losses[train_indices]
    nonmember_losses = losses[val_indices]

    print(f"Member loss stats: mean={member_losses.mean():.4f}, std={member_losses.std():.4f}")
    print(f"Non-member loss stats: mean={nonmember_losses.mean():.4f}, std={nonmember_losses.std():.4f}")

    tau, acc, tpr, tnr = best_threshold_from_losses(member_losses, nonmember_losses)

    # Balanced accuracy as in the R&D project: mean of TPR and TNR on a
    # hypothetical balanced set (50% members, 50% non-members).
    balanced_acc = 0.5 * (tpr + tnr)

    print("\n=== Membership Inference Attack (loss threshold) ===")
    print(f"Best threshold tau              = {tau:.4f}")
    print(f"Global attack accuracy          = {acc*100:.2f}% (on unbalanced 1600/400 set)")
    print(f"Balanced attack accuracy        = {balanced_acc*100:.2f}% (50% members / 50% non-members)")
    print(f"TPR (members correctly flagged) = {tpr*100:.2f}%")
    print(f"TNR (non-members correctly rejected) = {tnr*100:.2f}%")


if __name__ == "__main__":
    main()
