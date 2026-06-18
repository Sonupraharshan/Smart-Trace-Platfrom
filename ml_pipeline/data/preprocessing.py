"""
Dataset Preprocessing Pipeline
================================
- Parse Severstal train.csv
- Decode Run-Length Encoded (RLE) masks to binary numpy arrays
- Pivot per-class rows into per-image records with dominant label
- Create stratified train / val / test splits
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

from ml_pipeline.config import (
    TRAIN_CSV,
    TRAIN_IMAGES_DIR,
    ORIGINAL_IMG_HEIGHT,
    ORIGINAL_IMG_WIDTH,
    TRAIN_SPLIT,
    VAL_SPLIT,
    TEST_SPLIT,
)


# ──────────────────────────────────────────────
# RLE Decoding
# ──────────────────────────────────────────────
def rle_decode(rle_string: str, height: int = ORIGINAL_IMG_HEIGHT,
               width: int = ORIGINAL_IMG_WIDTH) -> np.ndarray:
    """
    Decode a run-length encoded string into a binary mask.

    The Severstal RLE format uses column-major (Fortran) ordering:
    pixels are numbered top-to-bottom, then left-to-right.

    Args:
        rle_string: Space-separated pairs of (start, length).
        height: Image height in pixels.
        width: Image width in pixels.

    Returns:
        Binary mask of shape (height, width) with dtype uint8.
    """
    if pd.isna(rle_string) or rle_string.strip() == "":
        return np.zeros((height, width), dtype=np.uint8)

    tokens = list(map(int, rle_string.split()))
    starts = tokens[0::2]
    lengths = tokens[1::2]

    mask_flat = np.zeros(height * width, dtype=np.uint8)
    for start, length in zip(starts, lengths):
        # RLE is 1-indexed
        mask_flat[start - 1: start - 1 + length] = 1

    # Column-major reshape then transpose to get (H, W)
    mask = mask_flat.reshape((height, width), order="F")
    return mask


def compute_mask_area_pct(rle_string: str, height: int = ORIGINAL_IMG_HEIGHT,
                          width: int = ORIGINAL_IMG_WIDTH) -> float:
    """Compute the percentage of the image covered by the mask."""
    mask = rle_decode(rle_string, height, width)
    total_pixels = height * width
    return (mask.sum() / total_pixels) * 100.0


# ──────────────────────────────────────────────
# CSV Parsing & Image-Level Label Assignment
# ──────────────────────────────────────────────
def load_and_prepare_dataframe(csv_path: Path = TRAIN_CSV,
                               images_dir: Path = TRAIN_IMAGES_DIR) -> pd.DataFrame:
    """
    Parse train.csv and produce a per-image DataFrame.

    Severstal CSV has one row per (ImageId, ClassId) pair.
    We pivot to one row per image with:
      - label: dominant defect class (largest mask area), or 0 if no defect
      - has_defect: boolean flag
      - mask_rle: RLE string for the dominant defect (or empty)
      - defect_area_pct: percentage of image area covered by dominant defect

    Returns:
        DataFrame with columns:
        [image_id, image_path, label, has_defect, mask_rle, defect_area_pct]
    """
    df = pd.read_csv(csv_path)
    df.columns = ["ImageId", "ClassId", "EncodedPixels"]

    # Compute mask area for every row
    df["mask_area_pct"] = df["EncodedPixels"].apply(
        lambda x: compute_mask_area_pct(x) if pd.notna(x) and str(x).strip() else 0.0
    )

    # Get all unique image IDs (including those with no defect)
    all_image_files = sorted([f.name for f in images_dir.glob("*.jpg")])

    # Build per-image records
    records = []
    for img_name in all_image_files:
        img_rows = df[df["ImageId"] == img_name]

        # Filter to rows that actually have a defect mask
        defect_rows = img_rows[img_rows["mask_area_pct"] > 0]

        if len(defect_rows) == 0:
            # No defect
            records.append({
                "image_id": img_name,
                "image_path": str(images_dir / img_name),
                "label": 0,
                "has_defect": False,
                "mask_rle": "",
                "defect_area_pct": 0.0,
            })
        else:
            # Pick dominant defect (largest mask area)
            dominant = defect_rows.loc[defect_rows["mask_area_pct"].idxmax()]
            records.append({
                "image_id": img_name,
                "image_path": str(images_dir / img_name),
                "label": int(dominant["ClassId"]),
                "has_defect": True,
                "mask_rle": dominant["EncodedPixels"],
                "defect_area_pct": dominant["mask_area_pct"],
            })

    result_df = pd.DataFrame(records)
    return result_df


# ──────────────────────────────────────────────
# Stratified Splits
# ──────────────────────────────────────────────
def create_splits(df: pd.DataFrame,
                  train_ratio: float = TRAIN_SPLIT,
                  val_ratio: float = VAL_SPLIT,
                  test_ratio: float = TEST_SPLIT,
                  random_state: int = 42):
    """
    Create stratified train / val / test splits.

    Args:
        df: DataFrame from load_and_prepare_dataframe()
        train_ratio: Fraction for training (default 0.70)
        val_ratio: Fraction for validation (default 0.15)
        test_ratio: Fraction for testing (default 0.15)
        random_state: Random seed for reproducibility

    Returns:
        (train_df, val_df, test_df) tuple of DataFrames
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"

    # First split: train vs (val + test)
    train_df, temp_df = train_test_split(
        df,
        test_size=(val_ratio + test_ratio),
        stratify=df["label"],
        random_state=random_state,
    )

    # Second split: val vs test
    relative_test = test_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test,
        stratify=temp_df["label"],
        random_state=random_state,
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print(f"[Splits] Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    print(f"[Splits] Train label dist:\n{train_df['label'].value_counts().sort_index()}")

    return train_df, val_df, test_df


# ──────────────────────────────────────────────
# Convenience: full pipeline
# ──────────────────────────────────────────────
def run_preprocessing():
    """Run the full preprocessing pipeline and return splits."""
    print("[Preprocessing] Loading and preparing dataset...")
    df = load_and_prepare_dataframe()
    print(f"[Preprocessing] Total images: {len(df)}")
    print(f"[Preprocessing] Defective: {df['has_defect'].sum()}, "
          f"Non-defective: {(~df['has_defect']).sum()}")
    print(f"[Preprocessing] Label distribution:\n{df['label'].value_counts().sort_index()}")

    train_df, val_df, test_df = create_splits(df)
    return train_df, val_df, test_df


if __name__ == "__main__":
    run_preprocessing()
