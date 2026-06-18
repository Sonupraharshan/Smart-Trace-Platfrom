"""
KPI Calculator — Industrial Quality Metrics
===============================================
Computes quality KPIs from inspection records stored in the database.
Designed to be called from Django views.
"""

from collections import Counter


def calculate_kpis(inspection_results: list) -> dict:
    """
    Calculate industrial quality KPIs from inspection results.

    Args:
        inspection_results: List of dicts (or Django model instances)
                            with keys/attrs: is_defective, predicted_class,
                            defect_label, severity_score, severity_category,
                            confidence

    Returns:
        Dictionary of KPIs:
        {
            total_inspected, defective_count, non_defective_count,
            defect_rate, pass_rate, rejection_rate,
            avg_severity, avg_confidence,
            most_common_defect, most_common_defect_count,
            quality_score, risk_score,
            defect_distribution, severity_distribution,
            class_percentages,
        }
    """
    if not inspection_results:
        return _empty_kpis()

    total = len(inspection_results)

    # Extract values (handle both dicts and ORM objects)
    def _get(obj, key, default=None):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    defective = [r for r in inspection_results if _get(r, "is_defective")]
    non_defective = [r for r in inspection_results if not _get(r, "is_defective")]

    defective_count = len(defective)
    non_defective_count = len(non_defective)

    defect_rate = (defective_count / total) * 100 if total > 0 else 0
    pass_rate = (non_defective_count / total) * 100 if total > 0 else 0
    rejection_rate = defect_rate  # In this context, defective = rejected

    # Severity metrics
    severity_scores = [
        _get(r, "severity_score", 0) for r in inspection_results
        if _get(r, "severity_score") is not None
    ]
    avg_severity = sum(severity_scores) / len(severity_scores) if severity_scores else 0

    # Confidence metrics
    confidences = [
        _get(r, "confidence", 0) for r in inspection_results
        if _get(r, "confidence") is not None
    ]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0

    # Defect distribution
    defect_labels = [_get(r, "defect_label", "Unknown") for r in defective]
    defect_counter = Counter(defect_labels)
    most_common = defect_counter.most_common(1)
    most_common_defect = most_common[0][0] if most_common else "N/A"
    most_common_defect_count = most_common[0][1] if most_common else 0

    # Full defect distribution
    defect_distribution = dict(defect_counter.most_common())

    # Class percentages (relative to defective images)
    class_percentages = {}
    if defective_count > 0:
        for label, count in defect_counter.items():
            class_percentages[label] = round((count / defective_count) * 100, 1)

    # Severity distribution
    severity_categories = [
        _get(r, "severity_category", "Unknown") for r in defective
    ]
    severity_distribution = dict(Counter(severity_categories))

    # Quality Score: 0-100, higher is better
    # Based on pass rate and average severity (lower severity = higher quality)
    quality_score = max(0, min(100, pass_rate - (avg_severity * 0.5)))

    # Risk Score: 0-100, higher is worse
    # Based on defect rate, severity, and volume
    risk_score = min(100, (defect_rate * 0.6) + (avg_severity * 2.0))

    return {
        "total_inspected": total,
        "defective_count": defective_count,
        "non_defective_count": non_defective_count,
        "defect_rate": round(defect_rate, 1),
        "pass_rate": round(pass_rate, 1),
        "rejection_rate": round(rejection_rate, 1),
        "avg_severity": round(avg_severity, 2),
        "avg_confidence": round(avg_confidence, 4),
        "most_common_defect": most_common_defect,
        "most_common_defect_count": most_common_defect_count,
        "quality_score": round(quality_score, 1),
        "risk_score": round(risk_score, 1),
        "defect_distribution": defect_distribution,
        "severity_distribution": severity_distribution,
        "class_percentages": class_percentages,
    }


def _empty_kpis() -> dict:
    """Return empty KPIs when no data is available."""
    return {
        "total_inspected": 0,
        "defective_count": 0,
        "non_defective_count": 0,
        "defect_rate": 0.0,
        "pass_rate": 0.0,
        "rejection_rate": 0.0,
        "avg_severity": 0.0,
        "avg_confidence": 0.0,
        "most_common_defect": "N/A",
        "most_common_defect_count": 0,
        "quality_score": 100.0,
        "risk_score": 0.0,
        "defect_distribution": {},
        "severity_distribution": {},
        "class_percentages": {},
    }
