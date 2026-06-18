"""
Centralized configuration for the ML pipeline.

All paths, hyperparameters, and constants are defined here
so every module draws from a single source of truth.
"""

import os
from pathlib import Path

# ──────────────────────────────────────────────
# Project root (two levels up from this file)
# ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent

# ──────────────────────────────────────────────
# Dataset paths
# ──────────────────────────────────────────────
DATASET_ROOT = BASE_DIR / "data" / "severstal-steel-defect-detection"
TRAIN_CSV = DATASET_ROOT / "train.csv"
TRAIN_IMAGES_DIR = DATASET_ROOT / "train_images"
TEST_IMAGES_DIR = DATASET_ROOT / "test_images"
FAISS_GALLERY_DIR = BASE_DIR / "data" / "faiss_gallery"

# ──────────────────────────────────────────────
# Model artifacts
# ──────────────────────────────────────────────
SAVED_MODELS_DIR = BASE_DIR / "saved_models"
MODEL_PATH = SAVED_MODELS_DIR / "resnet50_classifier.pth"
TRAINING_HISTORY_PATH = SAVED_MODELS_DIR / "training_history.json"

# ──────────────────────────────────────────────
# FAISS index
# ──────────────────────────────────────────────
FAISS_INDEX_DIR = BASE_DIR / "faiss_index"
FAISS_INDEX_PATH = FAISS_INDEX_DIR / "defect_index.faiss"
FAISS_METADATA_PATH = FAISS_INDEX_DIR / "index_metadata.json"

# ──────────────────────────────────────────────
# Media (user uploads & Grad-CAM outputs)
# ──────────────────────────────────────────────
MEDIA_ROOT = BASE_DIR / "media"
UPLOAD_DIR = MEDIA_ROOT / "uploads"
GRADCAM_OUTPUT_DIR = MEDIA_ROOT / "gradcam_outputs"
REPORTS_DIR = BASE_DIR / "reports"

# ──────────────────────────────────────────────
# Image constants (Severstal dataset)
# ──────────────────────────────────────────────
ORIGINAL_IMG_HEIGHT = 256
ORIGINAL_IMG_WIDTH = 1600
INPUT_IMG_SIZE = 256  # Resize target for model input

# ──────────────────────────────────────────────
# Class definitions
# ──────────────────────────────────────────────
NUM_CLASSES = 5  # 0=No Defect, 1-4=Defect Classes
CLASS_NAMES = {
    0: "No Defect",
    1: "Class 1 - Inclusion",
    2: "Class 2 - Patches",
    3: "Class 3 - Scratches",
    4: "Class 4 - Rolled-in Scale",
}

# Severity thresholds (percentage of image area)
SEVERITY_THRESHOLDS = {
    "low": (0, 5),
    "medium": (5, 15),
    "high": (15, 100),
}

# ──────────────────────────────────────────────
# Training hyperparameters
# ──────────────────────────────────────────────
BATCH_SIZE = 8
NUM_WORKERS = 4 if os.name != "nt" else 0  # Windows DataLoader workaround
NUM_EPOCHS = 25
LEARNING_RATE = 1e-4
FC_LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 5
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# ──────────────────────────────────────────────
# FAISS retrieval
# ──────────────────────────────────────────────
EMBEDDING_DIM = 2048  # ResNet50 avgpool output
FAISS_TOP_K = 5

# ──────────────────────────────────────────────
# ONNX model artifacts (for deployment / inference)
# ──────────────────────────────────────────────
ONNX_MODEL_PATH = SAVED_MODELS_DIR / "resnet50_classifier.onnx"
ONNX_FEATURES_PATH = SAVED_MODELS_DIR / "resnet50_features.onnx"
ONNX_ACTIVATIONS_PATH = SAVED_MODELS_DIR / "resnet50_activations.onnx"

# ──────────────────────────────────────────────
# Device
# ──────────────────────────────────────────────
try:
    import torch
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    # PyTorch not installed (e.g. Vercel deployment using ONNX Runtime)
    DEVICE = "cpu"

# ──────────────────────────────────────────────
# Ensure directories exist
# ──────────────────────────────────────────────
for d in [SAVED_MODELS_DIR, FAISS_INDEX_DIR, MEDIA_ROOT, UPLOAD_DIR,
          GRADCAM_OUTPUT_DIR, REPORTS_DIR, FAISS_GALLERY_DIR]:
    d.mkdir(parents=True, exist_ok=True)
