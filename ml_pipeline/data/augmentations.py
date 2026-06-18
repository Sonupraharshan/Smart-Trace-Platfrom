"""
Albumentations Augmentation Pipelines
=======================================
Separate train and val/test transforms.
Train: aggressive augmentation for robustness.
Val/Test: only resize + normalize for deterministic evaluation.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2

from ml_pipeline.config import INPUT_IMG_SIZE


def get_train_transforms(img_size: int = INPUT_IMG_SIZE) -> A.Compose:
    """
    Training augmentation pipeline.

    Includes geometric and photometric augmentations to improve
    model generalization on steel surface images.
    """
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


def get_val_transforms(img_size: int = INPUT_IMG_SIZE) -> A.Compose:
    """
    Validation / test transforms.

    Only resizing and normalization — no augmentation.
    """
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])
