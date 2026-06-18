"""
Augmentation & Preprocessing Pipelines
=========================================
Separate train and val/test transforms.

Training uses Albumentations with ToTensorV2 (requires PyTorch).
Inference provides a lightweight NumPy-only preprocessor that
works without PyTorch (for ONNX Runtime deployments).
"""

import cv2
import numpy as np

from ml_pipeline.config import INPUT_IMG_SIZE

# ImageNet normalisation constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ──────────────────────────────────────────────
# Lightweight preprocessor (NumPy only — no PyTorch)
# ──────────────────────────────────────────────
def preprocess_image_numpy(image: np.ndarray,
                           img_size: int = INPUT_IMG_SIZE) -> np.ndarray:
    """
    Preprocess an RGB image for ONNX model inference.

    Applies: resize → float32 → normalize → NCHW transpose.

    Args:
        image: RGB image as numpy array, shape (H, W, 3), dtype uint8
        img_size: Target square size

    Returns:
        Preprocessed array of shape (1, 3, img_size, img_size), dtype float32
    """
    # Resize
    resized = cv2.resize(image, (img_size, img_size),
                         interpolation=cv2.INTER_LINEAR)

    # Convert to float [0, 1]
    img = resized.astype(np.float32) / 255.0

    # Normalize with ImageNet stats
    img = (img - IMAGENET_MEAN) / IMAGENET_STD

    # HWC → CHW
    img = np.transpose(img, (2, 0, 1))

    # Add batch dimension
    return np.expand_dims(img, axis=0).astype(np.float32)


# ──────────────────────────────────────────────
# Albumentations pipelines (used by training code)
# Guarded behind try/except so they don't break
# deployments that lack PyTorch.
# ──────────────────────────────────────────────
def get_train_transforms(img_size: int = INPUT_IMG_SIZE):
    """
    Training augmentation pipeline.

    Includes geometric and photometric augmentations to improve
    model generalization on steel surface images.
    """
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.15,
            rotate_limit=15,
            border_mode=0,
            p=0.5,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.4,
        ),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.CoarseDropout(
            max_holes=8,
            max_height=img_size // 16,
            max_width=img_size // 16,
            fill_value=0,
            p=0.3,
        ),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


def get_val_transforms(img_size: int = INPUT_IMG_SIZE):
    """
    Validation / test transforms.

    Only resizing and normalization — no augmentation.
    """
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])
