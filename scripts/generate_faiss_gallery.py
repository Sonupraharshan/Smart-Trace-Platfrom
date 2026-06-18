"""
Generate deployable FAISS gallery images.

Creates compressed preview images for every image referenced by the FAISS
metadata. This keeps root-cause similar-image previews working in deployment
without committing the full Kaggle dataset.
"""

import json
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml_pipeline.config import (  # noqa: E402
    FAISS_GALLERY_DIR,
    FAISS_METADATA_PATH,
    TRAIN_IMAGES_DIR,
)


MAX_WIDTH = 640
JPEG_QUALITY = 70


def main():
    if not FAISS_METADATA_PATH.exists():
        raise FileNotFoundError(
            f"FAISS metadata not found: {FAISS_METADATA_PATH}. "
            "Run scripts/build_faiss_index.py first."
        )

    with open(FAISS_METADATA_PATH, "r") as f:
        metadata = json.load(f)

    image_ids = metadata.get("image_ids", [])
    FAISS_GALLERY_DIR.mkdir(parents=True, exist_ok=True)

    created = 0
    skipped = 0
    missing = 0

    for image_id in tqdm(image_ids, desc="Generating FAISS gallery"):
        source = TRAIN_IMAGES_DIR / image_id
        target = FAISS_GALLERY_DIR / image_id

        if target.exists():
            skipped += 1
            continue

        if not source.exists():
            missing += 1
            continue

        with Image.open(source) as img:
            img = img.convert("RGB")
            if img.width > MAX_WIDTH:
                ratio = MAX_WIDTH / img.width
                size = (MAX_WIDTH, max(1, int(img.height * ratio)))
                img = img.resize(size, Image.Resampling.LANCZOS)
            img.save(target, "JPEG", quality=JPEG_QUALITY, optimize=True)
            created += 1

    print(f"Created: {created}")
    print(f"Skipped existing: {skipped}")
    print(f"Missing source images: {missing}")
    print(f"Gallery path: {FAISS_GALLERY_DIR}")


if __name__ == "__main__":
    main()
