"""Shared constants for the Asset Health Index platform.

Centralizing these values avoids magic strings/numbers scattered across
modules and keeps business vocabulary consistent (health statuses,
feature names, recommendation text, dashboard color codes, etc.).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# BP Neural Network condition classes
# ---------------------------------------------------------------------------
CLASS_HEALTHY = "Healthy"
CLASS_SUB_HEALTHY = "Sub-Healthy"
CLASS_ABNORMAL = "Abnormal"
CLASS_FAULT = "Fault"

BP_CONDITION_CLASSES: tuple[str, ...] = (
    CLASS_HEALTHY,
    CLASS_SUB_HEALTHY,
    CLASS_ABNORMAL,
    CLASS_FAULT,
)

# ---------------------------------------------------------------------------
# Health status labels used by the classification engine (AHI-based)
# ---------------------------------------------------------------------------
HEALTH_STATUS_HEALTHY = "Healthy"
HEALTH_STATUS_SUB_HEALTHY = "Sub-Healthy"
HEALTH_STATUS_ABNORMAL = "Abnormal"
HEALTH_STATUS_FAULT = "Fault"

HEALTH_STATUSES: tuple[str, ...] = (
    HEALTH_STATUS_HEALTHY,
    HEALTH_STATUS_SUB_HEALTHY,
    HEALTH_STATUS_ABNORMAL,
    HEALTH_STATUS_FAULT,
)

# ---------------------------------------------------------------------------
# Recommendation text mapped 1:1 to health status
# ---------------------------------------------------------------------------
RECOMMENDATION_CONTINUE_NORMAL_OPERATION = "Continue Normal Operation"
RECOMMENDATION_INCREASE_MONITORING_FREQUENCY = "Increase Monitoring Frequency"
RECOMMENDATION_SCHEDULE_INSPECTION = "Schedule Inspection"
RECOMMENDATION_IMMEDIATE_MAINTENANCE_ASSESSMENT = "Immediate Maintenance Assessment"

RECOMMENDATION_BY_HEALTH_STATUS: dict[str, str] = {
    HEALTH_STATUS_HEALTHY: RECOMMENDATION_CONTINUE_NORMAL_OPERATION,
    HEALTH_STATUS_SUB_HEALTHY: RECOMMENDATION_INCREASE_MONITORING_FREQUENCY,
    HEALTH_STATUS_ABNORMAL: RECOMMENDATION_SCHEDULE_INSPECTION,
    HEALTH_STATUS_FAULT: RECOMMENDATION_IMMEDIATE_MAINTENANCE_ASSESSMENT,
}

# ---------------------------------------------------------------------------
# Dashboard color codes (Power BI / Dim_HealthStatus)
# ---------------------------------------------------------------------------
HEALTH_STATUS_COLOR_CODES: dict[str, str] = {
    HEALTH_STATUS_HEALTHY: "#2ECC71",       # Green
    HEALTH_STATUS_SUB_HEALTHY: "#F1C40F",   # Yellow
    HEALTH_STATUS_ABNORMAL: "#E67E22",      # Orange
    HEALTH_STATUS_FAULT: "#E74C3C",         # Red
}

HEALTH_STATUS_DESCRIPTIONS: dict[str, str] = {
    HEALTH_STATUS_HEALTHY: "Asset is operating within normal healthy parameters.",
    HEALTH_STATUS_SUB_HEALTHY: "Asset shows early signs of degradation; monitor more closely.",
    HEALTH_STATUS_ABNORMAL: "Asset condition is abnormal; inspection is recommended.",
    HEALTH_STATUS_FAULT: "Asset condition indicates a fault; immediate action required.",
}

# ---------------------------------------------------------------------------
# Operational measurement feature names (model input variables)
# ---------------------------------------------------------------------------
FEATURE_SPEED = "speed"
FEATURE_FREQUENCY = "frequency"
FEATURE_VOLTAGE = "voltage"
FEATURE_CURRENT = "current"
FEATURE_ACTIVE_POWER = "active_power"
FEATURE_REACTIVE_POWER = "reactive_power"
FEATURE_POWER_FACTOR = "power_factor"

INPUT_FEATURE_NAMES: tuple[str, ...] = (
    FEATURE_SPEED,
    FEATURE_FREQUENCY,
    FEATURE_VOLTAGE,
    FEATURE_CURRENT,
    FEATURE_ACTIVE_POWER,
    FEATURE_REACTIVE_POWER,
    FEATURE_POWER_FACTOR,
)

# ---------------------------------------------------------------------------
# Numeric defaults / bounds
# ---------------------------------------------------------------------------
DEFAULT_AHI_MIN = 0.0
DEFAULT_AHI_MAX = 100.0

DEFAULT_MODEL_VERSION = "v0.0.0-unreleased"
