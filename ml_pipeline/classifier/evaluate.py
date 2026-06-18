"""
Model Evaluation Module
=========================
Generates comprehensive classification metrics:
- Accuracy, Precision, Recall, F1 Score (per-class and macro)
- Confusion matrix (saved as image)
- Classification report
"""

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from tqdm import tqdm

from ml_pipeline.config import DEVICE, CLASS_NAMES, SAVED_MODELS_DIR


def evaluate_model(model, dataloader, device: torch.device = DEVICE) -> dict:
    """
    Evaluate the model on a dataset and return comprehensive metrics.

    Args:
        model: Trained PyTorch model
        dataloader: DataLoader for the evaluation set
        device: Device for inference

    Returns:
        Dictionary with keys:
        - accuracy, precision, recall, f1 (macro averages)
        - per_class_precision, per_class_recall, per_class_f1
        - confusion_matrix (numpy array)
        - classification_report (string)
    """
    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels, _ in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Compute metrics
    class_names = [CLASS_NAMES[i] for i in sorted(CLASS_NAMES.keys())]

    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, average="macro", zero_division=0),
        "recall": recall_score(all_labels, all_preds, average="macro", zero_division=0),
        "f1": f1_score(all_labels, all_preds, average="macro", zero_division=0),
        "per_class_precision": precision_score(
            all_labels, all_preds, average=None, zero_division=0
        ).tolist(),
        "per_class_recall": recall_score(
            all_labels, all_preds, average=None, zero_division=0
        ).tolist(),
        "per_class_f1": f1_score(
            all_labels, all_preds, average=None, zero_division=0
        ).tolist(),
        "confusion_matrix": confusion_matrix(all_labels, all_preds),
        "classification_report": classification_report(
            all_labels, all_preds, target_names=class_names, zero_division=0
        ),
    }

    # Print summary
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f} (macro)")
    print(f"Recall:    {metrics['recall']:.4f} (macro)")
    print(f"F1 Score:  {metrics['f1']:.4f} (macro)")
    print(f"\n{metrics['classification_report']}")

    return metrics


def plot_confusion_matrix(metrics: dict, save_path: str = None):
    """
    Plot and save the confusion matrix as an image.

    Args:
        metrics: Dict from evaluate_model() containing 'confusion_matrix'
        save_path: Path to save the figure (defaults to saved_models/confusion_matrix.png)
    """
    if save_path is None:
        save_path = str(SAVED_MODELS_DIR / "confusion_matrix.png")

    cm = metrics["confusion_matrix"]
    class_names = [CLASS_NAMES[i] for i in sorted(CLASS_NAMES.keys())]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        linewidths=0.5,
        linecolor="gray",
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title("Confusion Matrix — Steel Defect Classifier", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Evaluate] Confusion matrix saved to {save_path}")


def plot_training_history(history: dict, save_path: str = None):
    """
    Plot training/validation loss and accuracy curves.

    Args:
        history: Training history dict from train_model()
        save_path: Path to save the figure
    """
    if save_path is None:
        save_path = str(SAVED_MODELS_DIR / "training_history.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(history["train_loss"], label="Train Loss", color="#2196F3", linewidth=2)
    axes[0].plot(history["val_loss"], label="Val Loss", color="#FF5722", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(history["train_acc"], label="Train Acc", color="#4CAF50", linewidth=2)
    axes[1].plot(history["val_acc"], label="Val Acc", color="#9C27B0", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training & Validation Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Steel Defect Classifier — Training History", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Evaluate] Training history plot saved to {save_path}")
