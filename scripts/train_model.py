"""
Training Script
=================
CLI script to train the ResNet50 steel defect classifier.

Usage:
    python scripts/train_model.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_pipeline.data.preprocessing import run_preprocessing
from ml_pipeline.data.dataloader import create_dataloaders
from ml_pipeline.classifier.train import train_model
from ml_pipeline.classifier.evaluate import (
    evaluate_model,
    plot_confusion_matrix,
    plot_training_history,
)
from ml_pipeline.config import DEVICE


def main():
    print("=" * 60)
    print("Smart Quality Inspection — Model Training")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print()

    # Step 1: Preprocessing
    print("[1/4] Running preprocessing pipeline...")
    train_df, val_df, test_df = run_preprocessing()

    # Step 2: Create DataLoaders
    print("\n[2/4] Creating DataLoaders...")
    dataloaders = create_dataloaders(train_df, val_df, test_df)

    # Step 3: Train model
    print("\n[3/4] Training model...")
    model, history = train_model(dataloaders)

    # Step 4: Evaluate
    print("\n[4/4] Evaluating on test set...")
    metrics = evaluate_model(model, dataloaders["test"])
    plot_confusion_matrix(metrics)
    plot_training_history(history)

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Test Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Test F1 (macro): {metrics['f1']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
