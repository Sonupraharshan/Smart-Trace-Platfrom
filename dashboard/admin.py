"""Admin registration for dashboard models."""

from django.contrib import admin
from dashboard.models import InspectionResult, BatchReport, FAISSIndexMeta


@admin.register(InspectionResult)
class InspectionResultAdmin(admin.ModelAdmin):
    list_display = [
        "image_name", "defect_label", "confidence",
        "severity_category", "is_defective", "inspected_at",
    ]
    list_filter = ["is_defective", "severity_category", "defect_label"]
    search_fields = ["image_name"]


@admin.register(BatchReport)
class BatchReportAdmin(admin.ModelAdmin):
    list_display = [
        "batch_name", "total_images", "passed_images",
        "failed_images", "quality_score", "created_at",
    ]


@admin.register(FAISSIndexMeta)
class FAISSIndexMetaAdmin(admin.ModelAdmin):
    list_display = ["index_path", "num_vectors", "embedding_dim", "built_at"]
