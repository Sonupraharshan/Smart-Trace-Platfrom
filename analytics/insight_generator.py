"""
Automated Insight Generator
===============================
Produces human-readable insights from KPI data.
Rule-based engine that highlights anomalies, trends, and risks.
"""


def generate_insights(kpis: dict) -> list:
    """
    Generate automated textual insights from KPI data.

    Args:
        kpis: Dictionary from kpi_calculator.calculate_kpis()

    Returns:
        List of insight strings, ordered by priority (critical first).
    """
    insights = []

    if kpis["total_inspected"] == 0:
        return ["No inspection data available. Upload images to begin analysis."]

    # ── Defect Rate Insights ──
    defect_rate = kpis["defect_rate"]
    if defect_rate > 30:
        insights.append(
            f"🔴 CRITICAL: Defect rate is {defect_rate}% — "
            f"significantly above the 20% threshold. "
            f"Immediate investigation of the production line is recommended."
        )
    elif defect_rate > 20:
        insights.append(
            f"🟡 WARNING: Defect rate is {defect_rate}% — "
            f"approaching critical levels. "
            f"Schedule a maintenance review."
        )
    elif defect_rate > 10:
        insights.append(
            f"🟠 ATTENTION: Defect rate is {defect_rate}%. "
            f"Monitor closely for upward trends."
        )
    else:
        insights.append(
            f"🟢 Defect rate is {defect_rate}% — within acceptable range."
        )

    # ── Most Common Defect ──
    if kpis["most_common_defect"] != "N/A":
        pct = kpis["class_percentages"].get(kpis["most_common_defect"], 0)
        insights.append(
            f"📊 {kpis['most_common_defect']} is the most frequent defect type, "
            f"contributing {pct}% of all failures "
            f"({kpis['most_common_defect_count']} cases)."
        )

    # ── Severity Insights ──
    avg_severity = kpis["avg_severity"]
    if avg_severity > 15:
        insights.append(
            f"🔴 Average severity is HIGH ({avg_severity}%). "
            f"Multiple large-area defects detected. "
            f"This may indicate a systemic issue."
        )
    elif avg_severity > 5:
        insights.append(
            f"🟡 Average severity is MEDIUM ({avg_severity}%). "
            f"Some defects are covering significant surface area."
        )
    else:
        insights.append(
            f"🟢 Average severity is LOW ({avg_severity}%). "
            f"Production quality is stable."
        )

    # ── Quality Score ──
    quality = kpis["quality_score"]
    if quality >= 90:
        insights.append(
            f"⭐ Quality score: {quality}/100 — Excellent production quality."
        )
    elif quality >= 70:
        insights.append(
            f"📈 Quality score: {quality}/100 — Good, but room for improvement."
        )
    else:
        insights.append(
            f"⚠️ Quality score: {quality}/100 — Below standard. "
            f"Review defect patterns and process parameters."
        )

    # ── Risk Score ──
    risk = kpis["risk_score"]
    if risk >= 50:
        insights.append(
            f"🚨 Risk score: {risk}/100 — HIGH RISK. "
            f"Recommend halting batch and performing root cause analysis."
        )
    elif risk >= 25:
        insights.append(
            f"⚠️ Risk score: {risk}/100 — Moderate risk level."
        )

    # ── Defect Class Distribution ──
    dist = kpis.get("class_percentages", {})
    for defect_name, pct in dist.items():
        if pct >= 40:
            insights.append(
                f"📌 {defect_name} accounts for {pct}% of all defects — "
                f"this is a dominant failure mode that should be investigated."
            )

    # ── Volume Insight ──
    total = kpis["total_inspected"]
    if total >= 100:
        insights.append(
            f"📦 {total} images inspected. "
            f"Statistical confidence in metrics is high."
        )
    elif total >= 20:
        insights.append(
            f"📦 {total} images inspected. "
            f"Metrics are becoming statistically meaningful."
        )
    else:
        insights.append(
            f"📦 Only {total} images inspected so far. "
            f"Upload more images for reliable analytics."
        )

    # ── Severity Distribution ──
    sev_dist = kpis.get("severity_distribution", {})
    high_count = sev_dist.get("High", 0)
    if high_count > 0:
        high_pct = (high_count / kpis["defective_count"]) * 100 if kpis["defective_count"] > 0 else 0
        insights.append(
            f"⚡ {high_count} images ({high_pct:.0f}% of defects) have HIGH severity — "
            f"large defect areas requiring priority attention."
        )

    return insights
