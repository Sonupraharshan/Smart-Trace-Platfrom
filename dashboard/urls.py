"""URL routing for the dashboard app."""

from django.urls import path
from dashboard import views

app_name = "dashboard"

urlpatterns = [
    # Executive Dashboard (home page)
    path("", views.executive_dashboard, name="executive_dashboard"),

    # Inspection Workspace
    path("inspect/", views.inspection_workspace, name="inspection_workspace"),

    # Root Cause Analysis (for a specific inspection)
    path("root-cause/<int:inspection_id>/", views.root_cause_analysis, name="root_cause_analysis"),

    # Batch Report
    path("batch/", views.batch_report, name="batch_report"),
    path("batch/<int:batch_id>/", views.batch_detail, name="batch_detail"),
]
