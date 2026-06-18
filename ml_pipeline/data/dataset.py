"""
PyTorch Dataset for Severstal Steel Defect Detection
======================================================
Loads images, decodes RLE masks, applies augmentations,
and returns (image_tensor, label, mask) tuples.
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset

from ml_pipeline.data.preprocessing import rle_decode
from ml_pipeline.config import ORIGINAL_IMG_HEIGHT, ORIGINAL_IMG_WIDTH


class SteelDefectDataset(Dataset):
    """
    PyTorch Dataset for steel surface defect images.

    Each sample returns:
        image: Augmented image tensor (C, H, W)
        label: Integer class label (0-4)
        mask:  Binary defect mask as numpy array (H, W) — before resize
    """

    def __init__(self, dataframe: pd.DataFrame, transform=None):
        """
        Args:
            dataframe: DataFrame with columns
                       [image_id, image_path, label, has_defect, mask_rle, defect_area_pct]
            transform: Albumentations transform pipeline
        """
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int):
        row = self.dataframe.iloc[idx]

        # Load image (BGR → RGB)
        image_path = row["image_path"]
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Decode mask
        mask_rle = row["mask_rle"]
        if isinstance(mask_rle, str) and mask_rle.strip():
            mask = rle_decode(mask_rle, ORIGINAL_IMG_HEIGHT, ORIGINAL_IMG_WIDTH)
        else:
            mask = np.zeros(
                (ORIGINAL_IMG_HEIGHT, ORIGINAL_IMG_WIDTH), dtype=np.uint8
            )

        # Label
        label = int(row["label"])

        # Apply augmentations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return image, label, mask

    def get_image_path(self, idx: int) -> str:
        """Get the file path of an image by index."""
        return self.dataframe.iloc[idx]["image_path"]

    def get_image_id(self, idx: int) -> str:
        """Get the image filename by index."""
        return self.dataframe.iloc[idx]["image_id"]
