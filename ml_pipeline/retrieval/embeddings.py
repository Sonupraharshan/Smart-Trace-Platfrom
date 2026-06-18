"""
Feature Embedding Extraction
================================
Extracts 2048-dimensional feature vectors from the ResNet50 backbone
(avgpool layer, before the FC head) for all training images.

These embeddings are used by the FAISS index for similarity retrieval.
"""

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from ml_pipeline.config import DEVICE, EMBEDDING_DIM


def extract_embeddings(model, dataloader: DataLoader,
                       device: torch.device = DEVICE) -> tuple:
    """
    Extract feature embeddings from all images in a DataLoader.

    Uses the ResNet50 backbone up to avgpool (excluding FC head)
    to produce a 2048-dimensional feature vector per image.

    Args:
        model: Trained SteelDefectClassifier
        dataloader: DataLoader for the dataset
        device: Compute device

    Returns:
        Tuple of:
        - embeddings: numpy array of shape (N, 2048)
        - image_ids: list of image filenames
        - labels: numpy array of shape (N,)
    """
    model.eval()
    model.to(device)

    all_embeddings = []
    all_labels = []

    # Build feature extractor (everything except FC)
    import torch.nn as nn
    modules = list(model.backbone.children())[:-1]
    feature_extractor = nn.Sequential(*modules).to(device)

    with torch.no_grad():
        for images, labels, _ in tqdm(dataloader, desc="Extracting embeddings"):
            images = images.to(device)
            features = feature_extractor(images)
            features = torch.flatten(features, 1)  # (B, 2048)

            all_embeddings.append(features.cpu().numpy())
            all_labels.extend(labels.numpy())

    embeddings = np.vstack(all_embeddings).astype(np.float32)
    labels = np.array(all_labels)

    # Retrieve image IDs from the dataset
    image_ids = []
    dataset = dataloader.dataset
    for i in range(len(dataset)):
        image_ids.append(dataset.get_image_id(i))

    print(f"[Embeddings] Extracted {embeddings.shape[0]} vectors "
          f"of dim {embeddings.shape[1]}")

    return embeddings, image_ids, labels
