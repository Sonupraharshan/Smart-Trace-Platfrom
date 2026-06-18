"""Custom template tags and filters for the dashboard."""

import json
from django import template

register = template.Library()


@register.filter
def percentage(value, decimals=1):
    """Format a number as a percentage string."""
    try:
        return f"{float(value):.{decimals}f}%"
    except (ValueError, TypeError):
        return "0.0%"


@register.filter
def to_json(value):
    """Convert a Python object to a JSON string for use in JavaScript."""
    try:
        return json.dumps(value)
    except (TypeError, ValueError):
        return "{}"


@register.filter
def severity_color(category):
    """Return a CSS color class based on severity category."""
    colors = {
        "No Defect": "success",
        "Low": "info",
        "Medium": "warning",
        "High": "danger",
    }
    return colors.get(category, "secondary")


@register.filter
def confidence_color(confidence):
    """Return a CSS color based on confidence value."""
    try:
        conf = float(confidence)
        if conf >= 0.9:
            return "success"
        elif conf >= 0.7:
            return "warning"
        else:
            return "danger"
    except (ValueError, TypeError):
        return "secondary"


@register.filter
def multiply(value, arg):
    """Multiply the value by the argument."""
    try:
        return float(value) * float(arg)
    except (ValueError, TypeError):
        return 0
