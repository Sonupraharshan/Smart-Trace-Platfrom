"""
DataLoader Factory
====================
Creates PyTorch DataLoaders for train, validation, and test sets.
"""

from torch.utils.data import DataLoader
import pandas as pd

from ml_pipeline.data.dataset import SteelDefectDataset
from ml_pipeline.data.augmentations import get_train_transforms, get_val_transforms
from ml_pipeline.config import BATCH_SIZE, NUM_WORKERS


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame = None,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
) -> dict:
    """
    Create DataLoaders for each split.

    Args:
        train_df: Training split DataFrame
        val_df: Validation split DataFrame
        test_df: Test split DataFrame (optional)
        batch_size: Batch size for all loaders
        num_workers: Number of data loading workers

    Returns:
        Dict with keys 'train', 'val', and optionally 'test',
        each mapping to a DataLoader instance.
    """
    train_dataset = SteelDefectDataset(train_df, transform=get_train_transforms())
    val_dataset = SteelDefectDataset(val_df, transform=get_val_transforms())

    loaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
    }

    if test_df is not None:
        test_dataset = SteelDefectDataset(test_df, transform=get_val_transforms())
        loaders["test"] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    for name, loader in loaders.items():
        print(f"[DataLoader] {name}: {len(loader.dataset)} samples, "
              f"{len(loader)} batches")

    return loaders
