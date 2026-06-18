"""
FAISS Index Builder Script
============================
Extracts embeddings from all training images and builds the FAISS index.

Usage:
    python scripts/build_faiss_index.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_pipeline.data.preprocessing import run_preprocessing
from ml_pipeline.data.dataloader import create_dataloaders
from ml_pipeline.classifier.model import load_classifier
from ml_pipeline.retrieval.embeddings import extract_embeddings
from ml_pipeline.retrieval.index_builder import build_index
from ml_pipeline.config import DEVICE, MODEL_PATH


def main():
    print("=" * 60)
    print("Smart Quality Inspection — FAISS Index Builder")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print()

    # Step 1: Load model
    print("[1/3] Loading trained classifier...")
    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found at {MODEL_PATH}")
        print("Please run 'python scripts/train_model.py' first.")
        sys.exit(1)

    model = load_classifier(device=DEVICE)
    print("Model loaded successfully.")

    # Step 2: Prepare data
    print("\n[2/3] Preparing dataset...")
    train_df, val_df, test_df = run_preprocessing()
    dataloaders = create_dataloaders(train_df, val_df, test_df, batch_size=64)

    # Use full training set for index
    print("\n[3/3] Extracting embeddings and building index...")
    embeddings, image_ids, labels = extract_embeddings(
        model, dataloaders["train"], device=DEVICE
    )

    # Build FAISS index
    build_index(embeddings, image_ids, labels)

    print("\n" + "=" * 60)
    print("FAISS index built successfully!")
    print(f"Indexed {len(image_ids)} images")
    print("=" * 60)


if __name__ == "__main__":
    main()
