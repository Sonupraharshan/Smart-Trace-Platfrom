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
    Supports 1-click inspection for pre-loaded sample images.
    """
    form = ImageUploadForm()
    result = None

    if request.method == "POST":
        sample_choice = request.POST.get("sample_image_id")
        file_path = None
        display_name = ""

        if sample_choice:
            sample_images = _get_or_create_sample_images()
            if sample_choice == "sample_1" and len(sample_images) > 0:
                target_img = sample_images[0]
                display_name = "Sample_Defective_Steel.jpg"
            elif sample_choice == "sample_2" and len(sample_images) > 1:
                target_img = sample_images[1]
                display_name = "Sample_Clean_Steel.jpg"
            else:
                target_img = sample_images[0] if sample_images else None
                display_name = "Sample_Steel.jpg"

            if target_img and Path(target_img).exists():
                import shutil
                upload_dir = Path(settings.MEDIA_ROOT) / "uploads"
                upload_dir.mkdir(parents=True, exist_ok=True)
                filename = f"single_sample_{uuid.uuid4().hex[:8]}_{Path(target_img).name}"
                file_path = upload_dir / filename
                shutil.copy2(target_img, file_path)

        elif "image" in request.FILES:
            form = ImageUploadForm(request.POST, request.FILES)
            if form.is_valid():
                uploaded_file = request.FILES["image"]
                upload_dir = Path(settings.MEDIA_ROOT) / "uploads"
                upload_dir.mkdir(parents=True, exist_ok=True)
                filename = f"{uuid.uuid4().hex[:8]}_{uploaded_file.name}"
                file_path = upload_dir / filename
                display_name = uploaded_file.name
                with open(file_path, "wb+") as f:
                    for chunk in uploaded_file.chunks():
                        f.write(chunk)

        if file_path and Path(file_path).exists():
            try:
                from ml_pipeline.inference.engine import get_inference_engine
                engine = get_inference_engine()
                result = engine.inspect_image(str(file_path), generate_gradcam=True, find_similar=True)

                inspection = InspectionResult.objects.create(
                    image_name=display_name or Path(file_path).name,
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
                result["uploaded_image_url"] = f"{settings.MEDIA_URL}uploads/{Path(file_path).name}"

                if result.get("gradcam_filename"):
                    result["gradcam_url"] = (
                        f"{settings.MEDIA_URL}gradcam_outputs/{result['gradcam_filename']}"
                    )

                messages.success(request, f"Image '{display_name}' inspected successfully!")

            except Exception as e:
                messages.error(request, f"Inspection failed: {str(e)}")
                result = {"error": str(e)}

    sample_list = _sync_samples_to_media()
    context = {
        "form": form,
        "result": result,
        "sample_list": sample_list,
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
        is_sample_batch = request.POST.get("is_sample_batch") == "1"

        saved_paths = []
        upload_dir = Path(settings.MEDIA_ROOT) / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)

        if is_sample_batch and not files:
            import shutil
            sample_images = _get_or_create_sample_images()[:4]
            for img_path in sample_images:
                dest_filename = f"sample_{uuid.uuid4().hex[:8]}_{img_path.name}"
                dest_path = upload_dir / dest_filename
                shutil.copy2(img_path, dest_path)
                saved_paths.append(dest_path)
            files = saved_paths

        elif files:
            for f in files:
                filename = f"{uuid.uuid4().hex[:8]}_{f.name}"
                path = upload_dir / filename
                with open(path, "wb+") as dest:
                    for chunk in f.chunks():
                        dest.write(chunk)
                saved_paths.append(path)

        if saved_paths:
            batch_name = request.POST.get("batch_name", "").strip()
            if not batch_name:
                batch_name = "Sample_Lot_QA" if is_sample_batch else f"Batch_{uuid.uuid4().hex[:6]}"

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
    sample_list = _sync_samples_to_media()

    context = {
        "form": form,
        "batch_result": batch_result,
        "previous_batches": previous_batches,
        "sample_list": sample_list,
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


def _sync_samples_to_media():
    """Ensure sample images are available in media/samples for instant client previews."""
    import shutil
    dest_dir = Path(settings.MEDIA_ROOT) / "samples"
    dest_dir.mkdir(parents=True, exist_ok=True)
    raw_samples = _get_or_create_sample_images()
    synced = []
    labels = [
        "Defective (Pitted Surface)",
        "Clean Surface (Pass)",
        "Defective (Surface Scratch)",
        "Defective (Inclusion)",
        "Clean Surface (Pass)",
        "Defective (Patches)",
    ]
    for i, s_path in enumerate(raw_samples[:6]):
        dest_file = dest_dir / f"sample_{i+1}.jpg"
        if not dest_file.exists() and Path(s_path).exists():
            shutil.copy2(str(s_path), str(dest_file))
        size_kb = max(1, dest_file.stat().st_size // 1024) if dest_file.exists() else 50
        synced.append({
            "id": f"sample_{i+1}",
            "filename": f"Sample_Steel_Plate_0{i+1}.jpg",
            "url": f"{settings.MEDIA_URL}samples/sample_{i+1}.jpg",
            "size": f"{size_kb} KB",
            "dimensions": "1600 × 256 px",
            "label": labels[i] if i < len(labels) else f"Sample Plate #{i+1}",
            "is_defective": "Defective" in labels[i] if i < len(labels) else True,
        })
    return synced


def _get_or_create_sample_images():
    """
    Find existing sample images across standard dataset paths,
    or auto-generate 10 synthetic steel surface sample images if none exist.
    """
    base_dir = Path(settings.BASE_DIR)
    search_dirs = [
        base_dir / "data" / "sample_batch",
        base_dir / "data" / "severstal-steel-defect-detection" / "test_images",
        base_dir / "data" / "severstal-steel-defect-detection" / "train_images",
        Path(settings.MEDIA_ROOT) / "uploads",
    ]

    for d in search_dirs:
        if d.exists() and d.is_dir():
            found = set()
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
                found.update(d.glob(ext))
            if found:
                return sorted(list(found))[:10]


    # Fallback: Auto-generate 10 synthetic steel sample images in data/sample_batch
    target_dir = base_dir / "data" / "sample_batch"
    target_dir.mkdir(parents=True, exist_ok=True)

    try:
        import numpy as np
        from PIL import Image

        generated_images = []
        np.random.seed(42)
        for i in range(1, 11):
            filename = f"sample_steel_{i:02d}.jpg"
            filepath = target_dir / filename
            if not filepath.exists():
                base_color = np.random.randint(120, 180)
                noise = np.random.normal(0, 15, (256, 1600, 3))
                steel_img = np.clip(base_color + noise, 0, 255).astype(np.uint8)

                if i % 2 == 0:
                    y = np.random.randint(50, 200)
                    steel_img[y : y + 3, 200:800] = 30  # dark scratch defect

                img = Image.fromarray(steel_img)
                img.save(filepath, quality=90)

            generated_images.append(filepath)

        return generated_images
    except Exception as e:
        print(f"[Sample Batch] Fallback image generation warning: {e}")

    return []


def run_sample_batch(request):
    """
    Executes a pre-configured sample batch test using standard sample images.
    Auto-recovers and generates synthetic dataset if no files are found on disk.
    """
    import shutil

    sample_images = _get_or_create_sample_images()[:4]

    if not sample_images:
        messages.error(request, "Unable to locate or generate sample batch images.")
        return redirect("dashboard:batch_report")

    upload_dir = Path(settings.MEDIA_ROOT) / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Copy sample images to uploads directory so media URLs work as expected
    saved_paths = []
    for img_path in sample_images:
        dest_filename = f"sample_{uuid.uuid4().hex[:8]}_{img_path.name}"
        dest_path = upload_dir / dest_filename
        shutil.copy2(img_path, dest_path)
        saved_paths.append(dest_path)

    try:
        from ml_pipeline.inference.engine import get_inference_engine
        from dashboard.batch_processor import process_batch

        batch_name = f"Sample_Batch_{uuid.uuid4().hex[:6].upper()}"
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

        messages.success(
            request,
            f"Sample Batch '{batch_name}' processed successfully! "
            f"Inspected {batch_result['total_images']} images."
        )
        return redirect("dashboard:batch_detail", batch_id=report.id)

    except Exception as e:
        messages.error(request, f"Sample batch execution failed: {str(e)}")
        return redirect("dashboard:batch_report")


def download_sample_images(request):
    """
    Bundles available sample steel surface images into a ZIP archive for download.
    """
    import io
    import zipfile

    try:
        sample_images = _get_or_create_sample_images()
        if not sample_images:
            messages.error(request, "No sample images available to download.")
            return redirect("dashboard:batch_report")

        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for idx, img_path in enumerate(sample_images, start=1):
                if Path(img_path).exists():
                    arcname = f"sample_steel_{idx:02d}_{Path(img_path).name}"
                    zip_file.write(img_path, arcname=arcname)

        buffer.seek(0)
        response = HttpResponse(buffer.getvalue(), content_type="application/zip")
        response["Content-Disposition"] = 'attachment; filename="smarttrace_sample_steel_images.zip"'
        return response

    except Exception as e:
        messages.error(request, f"Could not generate sample ZIP: {str(e)}")
        return redirect("dashboard:batch_report")


def clear_history(request):
    """
    Clears all previous inspection results, batch reports, and temporary media files.
    """
    if request.method == "POST":
        try:
            inspection_count = InspectionResult.objects.count()
            batch_count = BatchReport.objects.count()

            InspectionResult.objects.all().delete()
            BatchReport.objects.all().delete()

            # Clean temporary files in media/uploads, media/gradcam_outputs, and reports
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
                            except Exception:
                                pass

            messages.success(
                request,
                f"Cleared all previous readings! Purged {inspection_count} inspections, "
                f"{batch_count} batch reports, and {deleted_files} files."
            )
        except Exception as e:
            messages.error(request, f"Failed to clear history: {str(e)}")

    return redirect("dashboard:executive_dashboard")




