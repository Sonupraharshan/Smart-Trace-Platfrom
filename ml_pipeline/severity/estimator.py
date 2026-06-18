"""
Defect Severity Estimation Module
====================================
Calculates severity from:
  1. Ground-truth RLE masks (when available during training/analysis)
  2. Grad-CAM activation maps (during inference, as proxy masks)

Severity Levels:
  Low:    0 – 5% of image area
  Medium: 5 – 15% of image area
  High:   15%+ of image area
"""

import numpy as np

from ml_pipeline.config import (
    SEVERITY_THRESHOLDS,
    ORIGINAL_IMG_HEIGHT,
    ORIGINAL_IMG_WIDTH,
)
from ml_pipeline.data.preprocessing import rle_decode


def estimate_severity_from_mask(mask: np.ndarray) -> dict:
    """
    Estimate defect severity from a binary mask.

    Args:
        mask: Binary mask of shape (H, W) with values 0 or 1.

    Returns:
        Dict with keys:
        - severity_score: Float (0-100) percentage of area affected
        - severity_category: 'Low', 'Medium', or 'High'
        - defect_area_pct: Same as severity_score (alias)
    """
    if mask is None or mask.size == 0:
        return {
            "severity_score": 0.0,
            "severity_category": "No Defect",
            "defect_area_pct": 0.0,
        }

    total_pixels = mask.shape[0] * mask.shape[1]
    defect_pixels = np.sum(mask > 0)
    area_pct = (defect_pixels / total_pixels) * 100.0

    category = _classify_severity(area_pct)

    return {
        "severity_score": round(area_pct, 2),
        "severity_category": category,
        "defect_area_pct": round(area_pct, 2),
    }


def estimate_severity_from_rle(rle_string: str,
                                height: int = ORIGINAL_IMG_HEIGHT,
                                width: int = ORIGINAL_IMG_WIDTH) -> dict:
    """
    Estimate severity from an RLE-encoded mask string.

    Args:
        rle_string: Run-length encoded mask
        height: Image height
        width: Image width

    Returns:
        Severity dict (same as estimate_severity_from_mask)
    """
    mask = rle_decode(rle_string, height, width)
    return estimate_severity_from_mask(mask)


def estimate_severity_from_gradcam(heatmap: np.ndarray,
                                    threshold: float = 0.5) -> dict:
    """
    Estimate severity using a Grad-CAM heatmap as a proxy mask.

    During inference we don't have ground-truth masks, so we threshold
    the Grad-CAM activation map to create a pseudo-mask.

    Args:
        heatmap: Grad-CAM heatmap of shape (H, W), values in [0, 1].
        threshold: Activation threshold above which a pixel is "defective".

    Returns:
        Severity dict
    """
    if heatmap is None or heatmap.size == 0:
        return {
            "severity_score": 0.0,
            "severity_category": "No Defect",
            "defect_area_pct": 0.0,
        }

    # Binarize heatmap
    binary_mask = (heatmap >= threshold).astype(np.uint8)
    return estimate_severity_from_mask(binary_mask)


def _classify_severity(area_pct: float) -> str:
    """Map area percentage to severity category."""
    if area_pct <= 0:
        return "No Defect"

    low_min, low_max = SEVERITY_THRESHOLDS["low"]
    med_min, med_max = SEVERITY_THRESHOLDS["medium"]

    if area_pct <= low_max:
        return "Low"
    elif area_pct <= med_max:
        return "Medium"
    else:
        return "High"
