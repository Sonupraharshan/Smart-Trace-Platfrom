"""
Dashboard Views
==================
All page views for the Smart Quality Inspection Platform.

Pages:
  1. Executive Dashboard — KPIs, charts, insights
  2. Inspection Workspace — single image upload + results
  3. Root Cause Analysis — similar defects from FAISS
  4. Batch Report — batch upload + aggregated report
"""

import os
import json
import uuid
from pathlib import Path

from django.shortcuts import render, get_object_or_404, redirect
from django.http import HttpResponse, FileResponse
from django.contrib import messages
from django.conf import settings

from dashboard.models import InspectionResult, BatchReport
from dashboard.forms import ImageUploadForm, BatchUploadForm
from analytics.kpi_calculator import calculate_kpis
from analytics.insight_generator import generate_insights


# ──────────────────────────────────────────────
# Executive Dashboard
# ──────────────────────────────────────────────
def executive_dashboard(request):
    """
    Main dashboard page with KPIs, charts, and insights.
    """
    # Get all inspection results
    inspections = InspectionResult.objects.all()
    inspection_list = list(inspections.values())

    # Calculate KPIs
    kpis = calculate_kpis(inspection_list)

    # Generate insights
    insights = generate_insights(kpis)

    # Recent inspections
    recent_inspections = inspections[:10]

    # Prepare chart data for Plotly
    defect_dist = kpis.get("defect_distribution", {})
    severity_dist = kpis.get("severity_distribution", {})

    context = {
        "kpis": kpis,
        "insights": insights,
        "recent_inspections": recent_inspections,
        "defect_dist_labels": list(defect_dist.keys()),
        "defect_dist_values": list(defect_dist.values()),
        "severity_dist_labels": list(severity_dist.keys()),
        "severity_dist_values": list(severity_dist.values()),
        "page_title": "Executive Dashboard",
    }

    return render(request, "dashboard/executive_dashboard.html", context)


# ──────────────────────────────────────────────
# Inspection Workspace
# ──────────────────────────────────────────────
def inspection_workspace(request):
    """
    Single image upload and inspection page.
    Shows prediction, confidence, severity, and Grad-CAM.
    """
    form = ImageUploadForm()
    result = None

    if request.method == "POST":
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = request.FILES["image"]

            # Save uploaded file
            upload_dir = Path(settings.MEDIA_ROOT) / "uploads"
            upload_dir.mkdir(parents=True, exist_ok=True)

            filename = f"{uuid.uuid4().hex[:8]}_{uploaded_file.name}"
            file_path = upload_dir / filename

            with open(file_path, "wb+") as f:
                for chunk in uploaded_file.chunks():
                    f.write(chunk)

            try:
                # Run inference
                from ml_pipeline.inference.engine import get_inference_engine
                engine = get_inference_engine()
                result = engine.inspect_image(str(file_path))

                # Save to database
                inspection = InspectionResult.objects.create(
                    image_name=uploaded_file.name,
                    image_path=str(file_path),
                    predicted_class=result["predicted_class"],
                    defect_label=result["defect_label"],
                    confidence=result["confidence"],
                    severity_score=result["severity_score"],
                    severity_category=result["severity_category"],
                    defect_area_pct=result["defect_area_pct"],
                    gradcam_path=result.get("gradcam_path", ""),
                    is_defective=result["is_defective"],
                )

                result["inspection_id"] = inspection.id
                result["uploaded_image_url"] = f"{settings.MEDIA_URL}uploads/{filename}"

                # Grad-CAM URL
                if result.get("gradcam_filename"):
                    result["gradcam_url"] = (
                        f"{settings.MEDIA_URL}gradcam_outputs/{result['gradcam_filename']}"
                    )

                messages.success(request, "Image inspected successfully!")

            except Exception as e:
                messages.error(request, f"Inspection failed: {str(e)}")
                result = {"error": str(e)}

    context = {
        "form": form,
        "result": result,
        "page_title": "Inspection Workspace",
    }

    return render(request, "dashboard/inspection_workspace.html", context)


# ──────────────────────────────────────────────
# Root Cause Analysis
# ──────────────────────────────────────────────
def root_cause_analysis(request, inspection_id):
    """
    Display similar historical defects for root cause analysis.
    """
    inspection = get_object_or_404(InspectionResult, id=inspection_id)

    # Re-run similarity search for this inspection
    similar_images = []
    try:
        from ml_pipeline.inference.engine import get_inference_engine
        engine = get_inference_engine()

        if engine.search_engine.is_loaded:
            # Extract embedding from the inspected image using the unified engine
            embedding = engine.extract_embedding(inspection.image_path)
            similar_images = engine.search_engine.find_similar(embedding)

            # Add media URLs for similar images
            for sim in similar_images:
                # Copy similar images to media for serving
                sim_path = Path(sim["image_path"])
                if sim_path.exists():
                    sim["image_url"] = f"{settings.MEDIA_URL}uploads/{sim['image_id']}"
                    # Ensure it can be served — copy to media if needed
                    dest = Path(settings.MEDIA_ROOT) / "uploads" / sim["image_id"]
                    if not dest.exists():
                        import shutil
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(str(sim_path), str(dest))

    except Exception as e:
        messages.warning(request, f"Similarity search unavailable: {str(e)}")

    # Prepare uploaded image URL
    uploaded_image_url = ""
    image_path = Path(inspection.image_path)
    if image_path.exists():
        uploaded_image_url = f"{settings.MEDIA_URL}uploads/{image_path.name}"

    # Grad-CAM URL
    gradcam_url = ""
    if inspection.gradcam_path:
        gradcam_name = Path(inspection.gradcam_path).name
        gradcam_url = f"{settings.MEDIA_URL}gradcam_outputs/{gradcam_name}"

    context = {
        "inspection": inspection,
        "similar_images": similar_images,
        "uploaded_image_url": uploaded_image_url,
        "gradcam_url": gradcam_url,
        "page_title": "Root Cause Analysis",
    }

    return render(request, "dashboard/root_cause_analysis.html", context)


# ──────────────────────────────────────────────
# Batch Report
# ──────────────────────────────────────────────
def batch_report(request):
    """
    Batch image upload and aggregated report page.
    """
    form = BatchUploadForm()
    batch_result = None

    if request.method == "POST":
        form = BatchUploadForm(request.POST, request.FILES)
        files = request.FILES.getlist("images")

        if files:
            batch_name = request.POST.get("batch_name", "").strip()
            if not batch_name:
                batch_name = f"Batch_{uuid.uuid4().hex[:6]}"

            # Save uploaded files
            upload_dir = Path(settings.MEDIA_ROOT) / "uploads"
            upload_dir.mkdir(parents=True, exist_ok=True)

            saved_paths = []
            for f in files:
                filename = f"{uuid.uuid4().hex[:8]}_{f.name}"
                path = upload_dir / filename
                with open(path, "wb+") as dest:
                    for chunk in f.chunks():
                        dest.write(chunk)
                saved_paths.append(path)

            try:
                from ml_pipeline.inference.engine import get_inference_engine
                from dashboard.batch_processor import process_batch

                engine = get_inference_engine()
                batch_result = process_batch(saved_paths, batch_name, engine)

                # Save batch report to DB
                report = BatchReport.objects.create(
                    batch_name=batch_name,
                    total_images=batch_result["total_images"],
                    passed_images=batch_result["passed_images"],
                    failed_images=batch_result["failed_images"],
                    avg_severity=batch_result["avg_severity"],
                    quality_score=batch_result["quality_score"],
                    defect_rate=batch_result["defect_rate"],
                    defect_distribution_json=json.dumps(
                        batch_result["defect_distribution"]
                    ),
                    report_path=batch_result.get("report_path", ""),
                )

                # Save individual inspection results linked to batch
                for r in batch_result["results"]:
                    if r.get("predicted_class", -1) >= 0:
                        InspectionResult.objects.create(
                            image_name=Path(r["image_path"]).name,
                            image_path=r["image_path"],
                            predicted_class=r["predicted_class"],
                            defect_label=r["defect_label"],
                            confidence=r["confidence"],
                            severity_score=r["severity_score"],
                            severity_category=r["severity_category"],
                            defect_area_pct=r.get("defect_area_pct", 0),
                            gradcam_path=r.get("gradcam_path", ""),
                            is_defective=r["is_defective"],
                            batch_report=report,
                        )

                batch_result["batch_id"] = report.id

                # Chart data
                batch_result["defect_dist_labels"] = list(
                    batch_result["defect_distribution"].keys()
                )
                batch_result["defect_dist_values"] = list(
                    batch_result["defect_distribution"].values()
                )
                batch_result["severity_dist_labels"] = list(
                    batch_result["severity_distribution"].keys()
                )
                batch_result["severity_dist_values"] = list(
                    batch_result["severity_distribution"].values()
                )

                messages.success(
                    request,
                    f"Batch '{batch_name}' processed: "
                    f"{batch_result['total_images']} images inspected."
                )

            except Exception as e:
                messages.error(request, f"Batch processing failed: {str(e)}")
                batch_result = {"error": str(e)}
        else:
            messages.warning(request, "No images uploaded.")

    # Get previous batch reports
    previous_batches = BatchReport.objects.all()[:10]

    context = {
        "form": form,
        "batch_result": batch_result,
        "previous_batches": previous_batches,
        "page_title": "Batch Inspection",
    }

    return render(request, "dashboard/batch_report.html", context)


def batch_detail(request, batch_id):
    """View details of a specific batch report."""
    report = get_object_or_404(BatchReport, id=batch_id)
    inspections = report.inspections.all()

    # KPIs for this batch
    inspection_list = list(inspections.values())
    kpis = calculate_kpis(inspection_list)
    insights = generate_insights(kpis)

    defect_dist = kpis.get("defect_distribution", {})
    severity_dist = kpis.get("severity_distribution", {})

    context = {
        "report": report,
        "inspections": inspections,
        "kpis": kpis,
        "insights": insights,
        "defect_dist_labels": list(defect_dist.keys()),
        "defect_dist_values": list(defect_dist.values()),
        "severity_dist_labels": list(severity_dist.keys()),
        "severity_dist_values": list(severity_dist.values()),
        "page_title": f"Batch Report: {report.batch_name}",
    }

    return render(request, "dashboard/batch_detail.html", context)
