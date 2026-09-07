"""
Clear Saved State Script
========================
Purges all historical InspectionResult and BatchReport records from the database
and deletes generated upload/gradcam/report files to restore a clean state.

Usage:
    python scripts/clear_saved_state.py
"""

import sys
import os
import shutil
from pathlib import Path
import django

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Setup Django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()

from django.conf import settings
from dashboard.models import InspectionResult, BatchReport

def clear_state():
    print("=" * 60)
    print("Smart Trace Platform — Resetting Saved State")
    print("=" * 60)

    # 1. Clear Database Records
    inspection_count = InspectionResult.objects.count()
    batch_count = BatchReport.objects.count()

    InspectionResult.objects.all().delete()
    BatchReport.objects.all().delete()

    print(f"Deleted {inspection_count} inspection records.")
    print(f"Deleted {batch_count} batch reports.")

    # 2. Clear Media Uploads (except sample_batch folder)
    media_root = Path(settings.MEDIA_ROOT)
    uploads_dir = media_root / "uploads"
    gradcam_dir = media_root / "gradcam_outputs"
    reports_dir = Path(settings.BASE_DIR) / "reports"

    deleted_files = 0

    for directory in [uploads_dir, gradcam_dir, reports_dir]:
        if directory.exists():
            for item in directory.glob("*"):
                if item.is_file():
                    try:
                        item.unlink()
                        deleted_files += 1
                    except Exception as e:
                        print(f"Warning deleting {item}: {e}")

    print(f"Cleared {deleted_files} temporary upload, Grad-CAM, and CSV report files.")
    print("=" * 60)
    print("State reset complete! Platform is now on a fresh clean slate.")
    print("=" * 60)

if __name__ == "__main__":
    clear_state()
