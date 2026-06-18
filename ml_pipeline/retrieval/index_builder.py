"""
FAISS Index Builder
=====================
Builds a FAISS index from precomputed feature embeddings.
Uses IndexFlatL2 (exact L2 distance search) which is
efficient enough for ~12K vectors.
"""

import json
import faiss
import numpy as np
from pathlib import Path

from ml_pipeline.config import (
    FAISS_INDEX_PATH,
    FAISS_METADATA_PATH,
    FAISS_INDEX_DIR,
    EMBEDDING_DIM,
)


def build_index(embeddings: np.ndarray,
                image_ids: list,
                labels: np.ndarray,
                index_path: str = None,
                metadata_path: str = None):
    """
    Build and save a FAISS index.

    Args:
        embeddings: Feature vectors of shape (N, 2048), float32
        image_ids: List of image filenames corresponding to each vector
        labels: Array of class labels for each vector
        index_path: Path to save the FAISS index file
        metadata_path: Path to save the metadata JSON
    """
    if index_path is None:
        index_path = str(FAISS_INDEX_PATH)
    if metadata_path is None:
        metadata_path = str(FAISS_METADATA_PATH)

    # Ensure float32
    embeddings = embeddings.astype(np.float32)

    # Normalize embeddings for cosine similarity via L2 distance
    faiss.normalize_L2(embeddings)

    # Build index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product on normalized = cosine similarity
    index.add(embeddings)

    # Save index
    FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, index_path)

    # Save metadata (image IDs + labels for each vector)
    metadata = {
        "num_vectors": int(embeddings.shape[0]),
        "embedding_dim": int(dim),
        "image_ids": image_ids,
        "labels": labels.tolist(),
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[FAISS] Index built: {embeddings.shape[0]} vectors, dim={dim}")
    print(f"[FAISS] Saved to {index_path}")
    print(f"[FAISS] Metadata saved to {metadata_path}")

    return index
