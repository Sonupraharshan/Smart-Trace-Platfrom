"""
Database Models for Quality Inspection Platform
===================================================
"""

import json
from django.db import models


class InspectionResult(models.Model):
    """Stores every single-image inspection result."""

    image_name = models.CharField(max_length=255)
    image_path = models.CharField(max_length=500)
    predicted_class = models.IntegerField(default=0)
    defect_label = models.CharField(max_length=100, default="No Defect")
    confidence = models.FloatField(default=0.0)
    severity_score = models.FloatField(default=0.0)
    severity_category = models.CharField(max_length=50, default="No Defect")
    defect_area_pct = models.FloatField(default=0.0)
    gradcam_path = models.CharField(max_length=500, blank=True, default="")
    is_defective = models.BooleanField(default=False)
    inspected_at = models.DateTimeField(auto_now_add=True)

    # Optional link to batch report
    batch_report = models.ForeignKey(
        "BatchReport",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="inspections",
    )

    class Meta:
        ordering = ["-inspected_at"]
        verbose_name = "Inspection Result"
        verbose_name_plural = "Inspection Results"

    def __str__(self):
        return f"{self.image_name} — {self.defect_label} ({self.confidence:.2%})"


class BatchReport(models.Model):
    """Aggregated batch-level metrics."""

    batch_name = models.CharField(max_length=255)
    total_images = models.IntegerField(default=0)
    passed_images = models.IntegerField(default=0)
    failed_images = models.IntegerField(default=0)
    avg_severity = models.FloatField(default=0.0)
    quality_score = models.FloatField(default=100.0)
    defect_rate = models.FloatField(default=0.0)
    defect_distribution_json = models.TextField(default="{}")
    report_path = models.CharField(max_length=500, blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "Batch Report"
        verbose_name_plural = "Batch Reports"

    def __str__(self):
        return f"Batch: {self.batch_name} ({self.total_images} images)"

    @property
    def defect_distribution(self):
        """Parse the JSON string into a dict."""
        try:
            return json.loads(self.defect_distribution_json)
        except (json.JSONDecodeError, TypeError):
            return {}

    @defect_distribution.setter
    def defect_distribution(self, value: dict):
        self.defect_distribution_json = json.dumps(value)


class FAISSIndexMeta(models.Model):
    """Tracks the current FAISS index state."""

    index_path = models.CharField(max_length=500)
    num_vectors = models.IntegerField(default=0)
    embedding_dim = models.IntegerField(default=2048)
    built_at = models.DateTimeField(auto_now_add=True)
    image_ids_json = models.TextField(default="[]")

    class Meta:
        verbose_name = "FAISS Index Metadata"
        verbose_name_plural = "FAISS Index Metadata"

    def __str__(self):
        return f"FAISS Index: {self.num_vectors} vectors"
