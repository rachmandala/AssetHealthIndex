from __future__ import annotations

RECOMMENDATIONS = {
    "Healthy": "Continue routine monitoring and preventive maintenance.",
    "Sub-Healthy": "Increase inspection frequency and plan condition-based maintenance.",
    "Abnormal": "Schedule maintenance intervention and perform root-cause diagnostics.",
    "Fault": "Initiate immediate corrective action and assess operational risk before restart.",
}


def recommendation_for_class(health_class: str) -> str:
    return RECOMMENDATIONS.get(health_class, "Review asset condition manually.")
