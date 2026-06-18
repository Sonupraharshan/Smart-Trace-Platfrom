"""
Batch Inspection Processor
=============================
Processes multiple images, aggregates results into a BatchReport,
and generates a downloadable CSV report.
"""

import csv
import json
from datetime import datetime
from pathlib import Path

from ml_pipeline.config import REPORTS_DIR


def process_batch(image_paths: list, batch_name: str, inference_engine) -> dict:
    """
    Run inspection on a batch of images.

    Args:
        image_paths: List of file paths to inspect
        batch_name: Name for this batch
        inference_engine: InferenceEngine instance

    Returns:
        Dictionary with batch results:
        {
            "batch_name": str,
            "total_images": int,
            "passed_images": int,
            "failed_images": int,
            "avg_severity": float,
            "quality_score": float,
            "defect_rate": float,
            "defect_distribution": dict,
            "severity_distribution": dict,
            "results": list[dict],  # Per-image results
            "report_path": str,
        }
    """
    results = []
    passed = 0
    failed = 0
    total_severity = 0.0
    defect_counts = {}
    severity_counts = {}

    for path in image_paths:
        try:
            result = inference_engine.inspect_image(str(path))
            results.append(result)

            if result["is_defective"]:
                failed += 1
                label = result["defect_label"]
                defect_counts[label] = defect_counts.get(label, 0) + 1
            else:
                passed += 1

            total_severity += result["severity_score"]

            sev_cat = result["severity_category"]
            severity_counts[sev_cat] = severity_counts.get(sev_cat, 0) + 1

        except Exception as e:
            print(f"[Batch] Error processing {path}: {e}")
            results.append({
                "image_path": str(path),
                "error": str(e),
                "is_defective": False,
                "predicted_class": -1,
                "defect_label": "Error",
                "confidence": 0.0,
                "severity_score": 0.0,
                "severity_category": "Error",
            })

    total = len(results)
    avg_severity = total_severity / total if total > 0 else 0.0
    quality_score = (passed / total * 100) if total > 0 else 100.0
    defect_rate = (failed / total * 100) if total > 0 else 0.0

    # Generate CSV report
    report_path = _generate_csv_report(results, batch_name)

    batch_result = {
        "batch_name": batch_name,
        "total_images": total,
        "passed_images": passed,
        "failed_images": failed,
        "avg_severity": round(avg_severity, 2),
        "quality_score": round(quality_score, 1),
        "defect_rate": round(defect_rate, 1),
        "defect_distribution": defect_counts,
        "severity_distribution": severity_counts,
        "results": results,
        "report_path": report_path,
    }

    return batch_result


def _generate_csv_report(results: list, batch_name: str) -> str:
    """Generate a CSV inspection report and return the file path."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in batch_name)
    filename = f"report_{safe_name}_{timestamp}.csv"
    report_path = REPORTS_DIR / filename

    with open(report_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Image", "Prediction", "Confidence",
            "Severity Score", "Severity Category",
            "Defective", "Defect Area %",
        ])

        for r in results:
            writer.writerow([
                Path(r.get("image_path", "")).name,
                r.get("defect_label", "Error"),
                f"{r.get('confidence', 0):.4f}",
                f"{r.get('severity_score', 0):.2f}",
                r.get("severity_category", "N/A"),
                "Yes" if r.get("is_defective") else "No",
                f"{r.get('defect_area_pct', 0):.2f}",
            ])

    print(f"[Batch] Report saved to {report_path}")
    return str(report_path)
