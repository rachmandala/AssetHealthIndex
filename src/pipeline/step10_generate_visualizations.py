"""Phase 2 - Step 10: Visualization exports (interactive HTML + Excel).

Reproduces the Han et al. (2023)-style diagnostic charts:

1. Training set health-state distribution (True vs. Predicted per sample)
2. Confusion matrix of the training set (with per-class recall/precision
   margins and overall accuracy, MATLAB ``plotconfusion``-style)
3. Test set health-state distribution (True vs. Predicted per sample)
4. Confusion matrix of the test set
5. Sample points vs Mahalanobis Distance (entire historical dataset)
6. Sample points vs Health Index (entire historical dataset)
7. Mahalanobis Distance vs Health Index
8. Sample points vs Health Score / AHI (entire historical dataset)

Outputs:
    data/exports/visualizations.html  (interactive Plotly dashboard)
    data/exports/visualizations.xlsx  (data tables + native Excel charts)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from openpyxl.chart import LineChart, Reference, ScatterChart, Series
from openpyxl.formatting.rule import ColorScaleRule
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix

from src.config.settings import get_settings
from src.ml.health_index import HealthIndexCalculator
from src.ml.mahalanobis import MahalanobisHealthModel
from src.pipeline.common import VISUALIZATIONS_EXCEL_PATH, VISUALIZATIONS_HTML_PATH
from src.preprocessing.preprocessor import Preprocessor
from src.utils.constants import BP_CONDITION_CLASSES
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Fixed condition-class order (matches Han et al. 2023 methodology), used for
# both the numeric state coding (1-4) and confusion matrix row/column order.
CLASS_ORDER = list(BP_CONDITION_CLASSES)


def _score_full_dataset(
    clean_df: pd.DataFrame,
    preprocessor: Preprocessor,
    baseline_model: MahalanobisHealthModel,
) -> pd.DataFrame:
    """Score every historical record (not just new measurements) for MD/HI/AHI.

    This reproduces the full-dataset scale used by the reference charts
    (sample index spanning the entire historical dataset).
    """
    X_all = preprocessor.transform(clean_df)
    distances = baseline_model.calculate_batch_distance(X_all)

    calculator = HealthIndexCalculator(settings=get_settings())
    hi, ahi = calculator.calculate_batch(distances)

    scores_df = clean_df[["timestamp", "asset_id", "label"]].reset_index(drop=True).copy()
    scores_df["mahalanobis_distance"] = distances
    scores_df["health_index"] = hi
    scores_df["ahi_score"] = ahi
    scores_df.insert(0, "sample_index", range(1, len(scores_df) + 1))
    return scores_df


def _state_distribution_df(
    y_true: np.ndarray, y_pred: np.ndarray, class_order: list[str]
) -> tuple[pd.DataFrame, float]:
    """Build a True-vs-Predicted numeric-state table for a train/test split."""
    code_map = {cls: i + 1 for i, cls in enumerate(class_order)}
    accuracy = float((y_true == y_pred).mean())
    df = pd.DataFrame(
        {
            "sample_index": range(1, len(y_true) + 1),
            "true_label": y_true,
            "predicted_label": y_pred,
            "true_state": [code_map[v] for v in y_true],
            "predicted_state": [code_map[v] for v in y_pred],
        }
    )
    return df, accuracy


def _confusion_summary(
    y_true: np.ndarray, y_pred: np.ndarray, class_order: list[str]
) -> dict:
    """Build a MATLAB ``plotconfusion``-style summary: counts + margins."""
    matrix = confusion_matrix(y_true, y_pred, labels=class_order)
    matrix_df = pd.DataFrame(matrix, index=class_order, columns=class_order)

    row_sums = matrix.sum(axis=1)
    col_sums = matrix.sum(axis=0)
    diag = matrix.diagonal()

    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.where(row_sums > 0, diag / row_sums * 100, 0.0)
        precision = np.where(col_sums > 0, diag / col_sums * 100, 0.0)
    accuracy = float(diag.sum() / matrix.sum() * 100) if matrix.sum() else 0.0

    return {
        "matrix_df": matrix_df,
        "recall_pct": recall,  # per true class (row margin)
        "precision_pct": precision,  # per predicted class (column margin)
        "accuracy_pct": accuracy,
        "class_order": class_order,
    }


def build_visualization_data(
    split_info: dict,
    clean_df: pd.DataFrame,
    preprocessor: Preprocessor,
    baseline_model: MahalanobisHealthModel,
) -> dict:
    """Assemble every dataset backing the 8 requested visualizations."""
    class_order = [c for c in CLASS_ORDER if c in set(split_info["labels"])]

    train_states, train_accuracy = _state_distribution_df(
        split_info["y_train"], split_info["y_train_pred"], class_order
    )
    test_states, test_accuracy = _state_distribution_df(
        split_info["y_test"], split_info["y_test_pred"], class_order
    )

    train_confusion = _confusion_summary(
        split_info["y_train"], split_info["y_train_pred"], class_order
    )
    test_confusion = _confusion_summary(
        split_info["y_test"], split_info["y_test_pred"], class_order
    )

    full_scores = _score_full_dataset(clean_df, preprocessor, baseline_model)

    return {
        "class_order": class_order,
        "train_states": train_states,
        "train_accuracy": train_accuracy,
        "test_states": test_states,
        "test_accuracy": test_accuracy,
        "train_confusion": train_confusion,
        "test_confusion": test_confusion,
        "full_scores": full_scores,
    }


def _add_state_distribution_trace(fig: go.Figure, states_df: pd.DataFrame, row: int, col: int) -> None:
    fig.add_trace(
        go.Scatter(
            x=states_df["sample_index"],
            y=states_df["true_state"],
            mode="lines+markers",
            name="True",
            marker=dict(color="red", symbol="circle", size=5),
            line=dict(color="red", width=1),
        ),
        row=row,
        col=col,
    )
    fig.add_trace(
        go.Scatter(
            x=states_df["sample_index"],
            y=states_df["predicted_state"],
            mode="lines+markers",
            name="Predicted",
            marker=dict(color="blue", symbol="circle-open", size=6),
            line=dict(color="blue", width=1),
        ),
        row=row,
        col=col,
    )


def _add_confusion_matrix_trace(fig: go.Figure, confusion: dict, row: int, col: int) -> None:
    """Render a MATLAB ``plotconfusion``-style matrix with recall/precision margins."""
    class_order = confusion["class_order"]
    n = len(class_order)
    matrix = confusion["matrix_df"].values
    recall = confusion["recall_pct"]
    precision = confusion["precision_pct"]
    accuracy = confusion["accuracy_pct"]

    size = n + 1
    text = np.full((size, size), "", dtype=object)
    colors = np.full((size, size), np.nan)  # 0 (incorrect/pink) -> 1 (correct/green)

    for i in range(n):
        for j in range(n):
            text[i, j] = f"{matrix[i, j]}"
            colors[i, j] = 1.0 if i == j else 0.0

    for i in range(n):
        text[i, n] = f"{recall[i]:.1f}%<br>{100 - recall[i]:.1f}%"
        colors[i, n] = recall[i] / 100.0
    for j in range(n):
        text[n, j] = f"{precision[j]:.1f}%<br>{100 - precision[j]:.1f}%"
        colors[n, j] = precision[j] / 100.0
    text[n, n] = f"{accuracy:.1f}%<br>{100 - accuracy:.1f}%"
    colors[n, n] = accuracy / 100.0

    y_labels = class_order + ["Recall %"]
    x_labels = class_order + ["Precision %"]

    fig.add_trace(
        go.Heatmap(
            z=colors,
            x=x_labels,
            y=y_labels,
            text=text,
            texttemplate="%{text}",
            colorscale=[[0.0, "#f8cfd1"], [1.0, "#a9d6a0"]],
            showscale=False,
            xgap=2,
            ygap=2,
        ),
        row=row,
        col=col,
    )
    fig.update_yaxes(autorange="reversed", row=row, col=col)


def _build_html_dashboard(data: dict) -> go.Figure:
    """Build a single interactive Plotly dashboard with all 8 charts."""
    fig = make_subplots(
        rows=4,
        cols=2,
        subplot_titles=(
            f"1. Training Set State Distribution (Accuracy={data['train_accuracy']:.2%})",
            "2. Confusion Matrix - Training Set",
            f"3. Test Set State Distribution (Accuracy={data['test_accuracy']:.2%})",
            "4. Confusion Matrix - Test Set",
            "5. Sample Points vs Mahalanobis Distance",
            "6. Sample Points vs Health Index",
            "7. Mahalanobis Distance vs Health Index",
            "8. Sample Points vs Health Score (AHI)",
        ),
        specs=[
            [{"type": "xy"}, {"type": "heatmap"}],
            [{"type": "xy"}, {"type": "heatmap"}],
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}],
        ],
        vertical_spacing=0.06,
        horizontal_spacing=0.12,
    )

    _add_state_distribution_trace(fig, data["train_states"], row=1, col=1)
    _add_confusion_matrix_trace(fig, data["train_confusion"], row=1, col=2)
    _add_state_distribution_trace(fig, data["test_states"], row=2, col=1)
    _add_confusion_matrix_trace(fig, data["test_confusion"], row=2, col=2)

    scores = data["full_scores"]
    fig.add_trace(
        go.Scatter(
            x=scores["sample_index"],
            y=scores["mahalanobis_distance"],
            mode="lines",
            line=dict(color="royalblue"),
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=scores["sample_index"],
            y=scores["health_index"],
            mode="lines",
            line=dict(color="royalblue"),
        ),
        row=3,
        col=2,
    )
    md_sorted = scores.sort_values("mahalanobis_distance")
    fig.add_trace(
        go.Scatter(
            x=md_sorted["mahalanobis_distance"],
            y=md_sorted["health_index"],
            mode="lines",
            line=dict(color="royalblue"),
        ),
        row=4,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=scores["sample_index"],
            y=scores["ahi_score"],
            mode="lines",
            line=dict(color="royalblue"),
        ),
        row=4,
        col=2,
    )

    fig.update_xaxes(title_text="Predict the Sample", row=1, col=1)
    fig.update_yaxes(title_text="Predict the Outcome", row=1, col=1, dtick=1)
    fig.update_xaxes(title_text="Predict the Sample", row=2, col=1)
    fig.update_yaxes(title_text="Predict the Outcome", row=2, col=1, dtick=1)
    fig.update_xaxes(title_text="Sample Points", row=3, col=1)
    fig.update_yaxes(title_text="Mahalanobis Distance", row=3, col=1)
    fig.update_xaxes(title_text="Sample Points", row=3, col=2)
    fig.update_yaxes(title_text="Health Index", row=3, col=2)
    fig.update_xaxes(title_text="Mahalanobis Distance", row=4, col=1)
    fig.update_yaxes(title_text="Health Index", row=4, col=1)
    fig.update_xaxes(title_text="Sample Points", row=4, col=2)
    fig.update_yaxes(title_text="Health Score (AHI)", row=4, col=2)

    fig.update_layout(
        height=1800,
        width=1500,
        showlegend=False,
        title_text="Asset Health Index - Model & Scoring Visualizations",
    )
    return fig


def _write_excel_workbook(data: dict, path) -> None:
    """Write data tables plus native Excel charts to a single workbook."""
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        data["train_states"].to_excel(writer, sheet_name="Train State Distribution", index=False)
        data["test_states"].to_excel(writer, sheet_name="Test State Distribution", index=False)

        for sheet_name, confusion in (
            ("Train Confusion Matrix", data["train_confusion"]),
            ("Test Confusion Matrix", data["test_confusion"]),
        ):
            df = confusion["matrix_df"].copy()
            df["Recall %"] = confusion["recall_pct"].round(1)
            precision_row = list(confusion["precision_pct"].round(1)) + [
                round(confusion["accuracy_pct"], 1)
            ]
            df.loc["Precision %"] = precision_row
            df.to_excel(writer, sheet_name=sheet_name)

        data["full_scores"].to_excel(writer, sheet_name="Full Dataset Scores", index=False)

        workbook = writer.book

        # --- Line charts: True vs Predicted state per sample ---
        for sheet_name, accuracy in (
            ("Train State Distribution", data["train_accuracy"]),
            ("Test State Distribution", data["test_accuracy"]),
        ):
            ws = workbook[sheet_name]
            n_rows = ws.max_row
            chart = LineChart()
            chart.title = f"{sheet_name} (Accuracy={accuracy:.2%})"
            chart.x_axis.title = "Sample Index"
            chart.y_axis.title = "State (1-4)"
            true_ref = Reference(ws, min_col=4, min_row=1, max_row=n_rows)
            pred_ref = Reference(ws, min_col=5, min_row=1, max_row=n_rows)
            categories_ref = Reference(ws, min_col=1, min_row=2, max_row=n_rows)
            chart.add_data(true_ref, titles_from_data=True)
            chart.add_data(pred_ref, titles_from_data=True)
            chart.set_categories(categories_ref)
            for series in chart.series:
                series.marker.symbol = "circle"
                series.graphicalProperties.line.width = 10000
            ws.add_chart(chart, "G2")

        # --- Conditional-formatted heatmaps for the confusion matrices ---
        for sheet_name, confusion in (
            ("Train Confusion Matrix", data["train_confusion"]),
            ("Test Confusion Matrix", data["test_confusion"]),
        ):
            n = len(confusion["class_order"])
            ws = workbook[sheet_name]
            last_col_letter = ws.cell(row=1, column=n + 2).column_letter
            cell_range = f"B2:{last_col_letter}{n + 2}"
            rule = ColorScaleRule(
                start_type="min", start_color="F8CFD1",
                end_type="max", end_color="A9D6A0",
            )
            ws.conditional_formatting.add(cell_range, rule)

        # --- Line/scatter charts: full-dataset MD / HI / AHI ---
        ws = workbook["Full Dataset Scores"]
        n_rows = ws.max_row
        header = [cell.value for cell in ws[1]]
        col = {name: header.index(name) + 1 for name in header}

        def _line_chart(title: str, y_col: str, y_title: str) -> LineChart:
            chart = LineChart()
            chart.title = title
            chart.x_axis.title = "Sample Points"
            chart.y_axis.title = y_title
            y_values = Reference(ws, min_col=col[y_col], min_row=1, max_row=n_rows)
            categories_ref = Reference(ws, min_col=col["sample_index"], min_row=2, max_row=n_rows)
            chart.add_data(y_values, titles_from_data=True)
            chart.set_categories(categories_ref)
            return chart

        def _scatter_chart(title: str, y_col: str, x_col: str, y_title: str, x_title: str) -> ScatterChart:
            chart = ScatterChart()
            chart.title = title
            chart.x_axis.title = x_title
            chart.y_axis.title = y_title
            x_values = Reference(ws, min_col=col[x_col], min_row=2, max_row=n_rows)
            y_values = Reference(ws, min_col=col[y_col], min_row=1, max_row=n_rows)
            series = Series(y_values, x_values, title_from_data=True)
            series.marker.symbol = "none"
            chart.series.append(series)
            return chart

        ws.add_chart(
            _line_chart(
                "5. Sample Points vs Mahalanobis Distance",
                "mahalanobis_distance",
                "Mahalanobis Distance",
            ),
            "J2",
        )
        ws.add_chart(
            _line_chart("6. Sample Points vs Health Index", "health_index", "Health Index"),
            "J20",
        )
        ws.add_chart(
            _scatter_chart(
                "7. Mahalanobis Distance vs Health Index",
                "health_index",
                "mahalanobis_distance",
                "Health Index",
                "Mahalanobis Distance",
            ),
            "J38",
        )
        ws.add_chart(
            _line_chart(
                "8. Sample Points vs Health Score (AHI)", "ahi_score", "Health Score (AHI)"
            ),
            "J56",
        )


def run_visualization_export(
    split_info: dict,
    clean_df: pd.DataFrame,
    preprocessor: Preprocessor,
    baseline_model: MahalanobisHealthModel,
) -> dict:
    """Build and export all 8 visualizations as interactive HTML + Excel.

    Args:
        split_info: ``split_info`` dict returned by Step 4.
        clean_df: Cleaned historical operational measurements (Step 2).
        preprocessor: Fitted :class:`Preprocessor` (Step 2).
        baseline_model: Fitted :class:`MahalanobisHealthModel` (Step 3).

    Returns:
        Mapping of ``{"html": path, "excel": path}``.
    """
    data = build_visualization_data(split_info, clean_df, preprocessor, baseline_model)

    VISUALIZATIONS_HTML_PATH.parent.mkdir(parents=True, exist_ok=True)
    dashboard = _build_html_dashboard(data)
    dashboard.write_html(VISUALIZATIONS_HTML_PATH, include_plotlyjs="cdn")
    logger.info("Wrote interactive HTML dashboard to '%s'.", VISUALIZATIONS_HTML_PATH)

    VISUALIZATIONS_EXCEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    _write_excel_workbook(data, VISUALIZATIONS_EXCEL_PATH)
    logger.info("Wrote Excel visualization workbook to '%s'.", VISUALIZATIONS_EXCEL_PATH)

    return {"html": VISUALIZATIONS_HTML_PATH, "excel": VISUALIZATIONS_EXCEL_PATH}


if __name__ == "__main__":
    from src.pipeline.step2_preprocess_data import run_preprocessing
    from src.pipeline.step3_build_healthy_baseline import run_baseline_creation
    from src.pipeline.step4_train_bp_network import run_bp_training

    _clean_df, _preprocessor = run_preprocessing()
    _baseline_model = run_baseline_creation(_clean_df, _preprocessor)
    _, _metrics, _split_info = run_bp_training(_clean_df, _preprocessor)
    run_visualization_export(_split_info, _clean_df, _preprocessor, _baseline_model)
