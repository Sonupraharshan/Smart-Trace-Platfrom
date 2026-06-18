"""
Training Loop for Steel Defect Classifier
============================================
- CrossEntropy with class weights (imbalanced dataset)
- AdamW optimizer with differential learning rates
- ReduceLROnPlateau scheduler
- Early stopping
- Checkpoint best model
- Save training history to JSON
"""

import json
import time
import torch
import torch.nn as nn
import numpy as np
from collections import Counter
from pathlib import Path
from tqdm import tqdm

from ml_pipeline.classifier.model import SteelDefectClassifier
from ml_pipeline.config import (
    DEVICE,
    NUM_CLASSES,
    NUM_EPOCHS,
    EARLY_STOPPING_PATIENCE,
    WEIGHT_DECAY,
    MODEL_PATH,
    TRAINING_HISTORY_PATH,
    SAVED_MODELS_DIR,
)


def compute_class_weights(dataloader) -> torch.Tensor:
    """
    Compute inverse-frequency class weights from a DataLoader.

    This helps the model pay more attention to rare defect classes.
    """
    if hasattr(dataloader.dataset, "dataframe"):
        label_counts = Counter(dataloader.dataset.dataframe["label"].astype(int))
    else:
        label_counts = Counter()
        for _, labels, _ in dataloader:
            for label in labels.numpy():
                label_counts[label] += 1

    total = sum(label_counts.values())
    weights = []
    for i in range(NUM_CLASSES):
        count = label_counts.get(i, 1)
        weights.append(total / (NUM_CLASSES * count))

    weights = torch.FloatTensor(weights)
    # Normalize so max weight = 3.0 to prevent instability
    weights = weights / weights.max() * 3.0
    return weights


def train_model(
    dataloaders: dict,
    num_epochs: int = NUM_EPOCHS,
    patience: int = EARLY_STOPPING_PATIENCE,
    device: torch.device = DEVICE,
) -> tuple:
    """
    Train the steel defect classifier.

    Args:
        dataloaders: Dict with 'train' and 'val' DataLoaders
        num_epochs: Maximum training epochs
        patience: Early stopping patience (epochs without improvement)
        device: Training device (cuda/cpu)

    Returns:
        (trained_model, history_dict)
    """
    print(f"[Train] Device: {device}")
    print(f"[Train] Epochs: {num_epochs}, Patience: {patience}")

    # Initialize model
    model = SteelDefectClassifier(pretrained=True)
    model.to(device)

    # Compute class weights from training data
    print("[Train] Computing class weights...")
    class_weights = compute_class_weights(dataloaders["train"]).to(device)
    print(f"[Train] Class weights: {class_weights.cpu().numpy()}")

    # Loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer with differential learning rates
    param_groups = model.get_trainable_params()
    param_groups[0]["weight_decay"] = WEIGHT_DECAY
    param_groups[1]["weight_decay"] = WEIGHT_DECAY
    optimizer = torch.optim.AdamW(param_groups)

    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    # Training history
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "learning_rates": [],
        "epoch_times": [],
    }

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        epoch_start = time.time()
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"{'='*60}")

        # ── Training phase ──
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        train_loader = dataloaders["train"]
        pbar = tqdm(train_loader, desc="Training", leave=False)

        for images, labels, _ in pbar:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{train_correct / train_total:.4f}"
            })

        train_loss /= train_total
        train_acc = train_correct / train_total

        # ── Validation phase ──
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels, _ in dataloaders["val"]:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        # Update scheduler
        scheduler.step(val_loss)

        epoch_time = time.time() - epoch_start

        # Record history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["learning_rates"].append(optimizer.param_groups[0]["lr"])
        history["epoch_times"].append(epoch_time)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        print(f"Epoch Time: {epoch_time:.1f}s")

        # ── Checkpoint best model ──
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), str(MODEL_PATH))
            print(f"✓ Saved best model (val_loss={val_loss:.4f})")
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement}/{patience} epochs")

        # ── Early stopping ──
        if epochs_without_improvement >= patience:
            print(f"\n[Early Stopping] No improvement for {patience} epochs. Stopping.")
            break

    # Save training history
    with open(str(TRAINING_HISTORY_PATH), "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n[Train] History saved to {TRAINING_HISTORY_PATH}")

    # Load best model
    model.load_state_dict(
        torch.load(str(MODEL_PATH), map_location=device, weights_only=True)
    )
    print(f"[Train] Best model loaded from {MODEL_PATH}")

    return model, history
