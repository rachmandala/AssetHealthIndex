"""Phase 2 - Step 6: Health Index (HI) and Asset Health Index (AHI) calculation.

Converts Mahalanobis Distance scores (Step 5) into bounded Health Index and
Asset Health Index scores using the ``b`` parameter from configuration.
"""

from __future__ import annotations

import pandas as pd

from src.config.settings import get_settings
from src.ml.health_index import HealthIndexCalculator
from src.pipeline.common import AHI_SCORES_PATH
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_health_index_calculation(md_scores_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate HI/AHI for each scored record and write ``ahi_scores.csv``.

    Args:
        md_scores_df: Output of Step 5 with a ``mahalanobis_distance`` column.

    Returns:
        DataFrame with ``asset_id``, ``timestamp``, ``mahalanobis_distance``,
        ``health_index``, ``ahi_score``.
    """
    settings = get_settings()
    calculator = HealthIndexCalculator(settings=settings)

    hi, ahi = calculator.calculate_batch(md_scores_df["mahalanobis_distance"])

    result_df = md_scores_df.copy()
    result_df["health_index"] = hi
    result_df["ahi_score"] = ahi

    AHI_SCORES_PATH.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(AHI_SCORES_PATH, index=False)
    logger.info(
        "Wrote %d AHI scores to '%s' (b=%.4f).",
        len(result_df),
        AHI_SCORES_PATH,
        calculator.b,
    )

    return result_df


if __name__ == "__main__":
    from src.pipeline.step5_score_mahalanobis import run_mahalanobis_scoring

    _md_scores_df = run_mahalanobis_scoring()
    run_health_index_calculation(_md_scores_df)
