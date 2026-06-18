"""
Similarity Search Engine
==========================
Queries the FAISS index with a new image embedding
and returns the top-k most similar historical defect cases.
"""

import json
import faiss
import numpy as np
from pathlib import Path

from ml_pipeline.config import (
    FAISS_INDEX_PATH,
    FAISS_METADATA_PATH,
    FAISS_TOP_K,
    CLASS_NAMES,
    FAISS_GALLERY_DIR,
    TRAIN_IMAGES_DIR,
)


class SimilaritySearchEngine:
    """
    Retrieves similar historical defects using FAISS.

    The index uses cosine similarity (inner product on normalized vectors).
    Higher scores = more similar.
    """

    def __init__(self, index_path: str = None, metadata_path: str = None):
        """Load the FAISS index and metadata from disk."""
        if index_path is None:
            index_path = str(FAISS_INDEX_PATH)
        if metadata_path is None:
            metadata_path = str(FAISS_METADATA_PATH)

        self.index = None
        self.metadata = None
        self._loaded = False

        if Path(index_path).exists() and Path(metadata_path).exists():
            self.index = faiss.read_index(index_path)
            with open(metadata_path, "r") as f:
                self.metadata = json.load(f)
            self._loaded = True
            print(f"[Search] Loaded FAISS index: "
                  f"{self.metadata['num_vectors']} vectors")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def find_similar(self, query_embedding: np.ndarray,
                     top_k: int = FAISS_TOP_K) -> list:
        """
        Find the top-k most similar images to the query.

        Args:
            query_embedding: Feature vector of shape (1, 2048) or (2048,)
            top_k: Number of similar images to retrieve

        Returns:
            List of dicts with keys:
            - image_id: filename of the similar image
            - image_path: full path to the similar image
            - similarity_score: cosine similarity (0-1)
            - label: defect class of the similar image
            - label_name: human-readable class name
        """
        if not self._loaded:
            print("[Search] Warning: FAISS index not loaded. Returning empty results.")
            return []

        # Reshape and normalize
        query = query_embedding.astype(np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)
        faiss.normalize_L2(query)

        # Search
        scores, indices = self.index.search(query, top_k)

        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < 0:
                continue  # Invalid index

            image_id = self.metadata["image_ids"][idx]
            label = self.metadata["labels"][idx]
            gallery_path = FAISS_GALLERY_DIR / image_id
            if gallery_path.exists():
                image_path = str(gallery_path)
            else:
                image_path = str(TRAIN_IMAGES_DIR / image_id)

            results.append({
                "rank": i + 1,
                "image_id": image_id,
                "image_path": image_path,
                "similarity_score": round(float(score), 4),
                "label": label,
                "label_name": CLASS_NAMES.get(label, f"Class {label}"),
            })

        return results
