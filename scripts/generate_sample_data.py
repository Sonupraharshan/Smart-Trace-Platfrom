"""
Generate Sample Data
======================
Seeds the Django database with sample inspection records
for demo purposes (so the dashboard has data to display).

Usage:
    python scripts/generate_sample_data.py
"""

import sys
import os
import random
import django

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Django setup
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()

from dashboard.models import InspectionResult, BatchReport
from ml_pipeline.config import CLASS_NAMES, TRAIN_IMAGES_DIR


def main():
    print("=" * 60)
    print("Smart Quality Inspection — Sample Data Generator")
    print("=" * 60)

    # Confirm
    existing = InspectionResult.objects.count()
    if existing > 0:
        print(f"Database already has {existing} inspection records.")
        response = input("Add more sample data? (y/N): ").strip().lower()
        if response != "y":
            print("Aborted.")
            return

    # Get some real image names from the dataset
    image_files = sorted([f.name for f in TRAIN_IMAGES_DIR.glob("*.jpg")])[:200]

    if not image_files:
        print("ERROR: No images found in training set.")
        return

    print(f"Generating sample inspections from {len(image_files)} images...")

    severity_options = [
        ("No Defect", 0.0),
        ("Low", random.uniform(0.5, 4.9)),
        ("Medium", random.uniform(5.1, 14.5)),
        ("High", random.uniform(15.1, 35.0)),
    ]

    records = []
    for img_name in image_files:
        # Random defect assignment (weighted towards no-defect)
        if random.random() < 0.4:
            # No defect
            label = 0
            confidence = random.uniform(0.75, 0.99)
            severity_cat = "No Defect"
            severity_score = 0.0
        else:
            # Defect
            label = random.choice([1, 2, 3, 4])
            confidence = random.uniform(0.55, 0.98)
            severity_cat = random.choice(["Low", "Medium", "High"])
            if severity_cat == "Low":
                severity_score = round(random.uniform(0.5, 4.9), 2)
            elif severity_cat == "Medium":
                severity_score = round(random.uniform(5.1, 14.5), 2)
            else:
                severity_score = round(random.uniform(15.1, 35.0), 2)

        records.append(InspectionResult(
            image_name=img_name,
            image_path=str(TRAIN_IMAGES_DIR / img_name),
            predicted_class=label,
            defect_label=CLASS_NAMES.get(label, f"Class {label}"),
            confidence=round(confidence, 4),
            severity_score=severity_score,
            severity_category=severity_cat,
            defect_area_pct=severity_score,
            is_defective=(label != 0),
        ))

    # Bulk create
    InspectionResult.objects.bulk_create(records)
    print(f"Created {len(records)} sample inspection records.")

    # Create a sample batch report
    defective = [r for r in records if r.is_defective]
    non_defective = [r for r in records if not r.is_defective]

    BatchReport.objects.create(
        batch_name="Sample Production Batch",
        total_images=len(records),
        passed_images=len(non_defective),
        failed_images=len(defective),
        avg_severity=round(sum(r.severity_score for r in records) / len(records), 2),
        quality_score=round(len(non_defective) / len(records) * 100, 1),
        defect_rate=round(len(defective) / len(records) * 100, 1),
        defect_distribution_json="{}",
    )
    print("Created sample batch report.")

    print("\n" + "=" * 60)
    print("Sample data generation complete!")
    print(f"Total records: {InspectionResult.objects.count()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
