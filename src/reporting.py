"""Automated PDF reporting for the crypto sentiment analysis project."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis import AdvancedAnalyticsResults, AnalysisResults
from src.utils import REPORTS_DIR


LOGGER = logging.getLogger(__name__)


def _format_value(value: object) -> str:
    """Format table values for report display."""
    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        if np.isinf(value):
            return "inf"
        return f"{value:,.4g}"
    return str(value)


def _table_data(
    dataframe: pd.DataFrame,
    max_rows: int = 8,
    max_columns: int = 8,
) -> list[list[str]]:
    """Convert a DataFrame into ReportLab table data."""
    if dataframe.empty:
        return [["No data available"]]
    preview = dataframe.head(max_rows).iloc[:, :max_columns]
    preview = preview.replace([np.inf, -np.inf], np.nan)
    return [
        [str(column) for column in preview.columns],
        *[
            [_format_value(value) for value in row]
            for row in preview.itertuples(index=False, name=None)
        ],
    ]


def _find_best_sentiment(analysis_results: AnalysisResults) -> str:
    """Summarize the strongest sentiment regime by average PnL."""
    data = analysis_results.sentiment_profitability
    if data.empty or "avg_pnl" not in data.columns:
        return "Sentiment profitability could not be ranked from the available data."
    best = data.sort_values("avg_pnl", ascending=False).iloc[0]
    return (
        f"The strongest average trade PnL appears during {best['classification']} "
        f"with average PnL of {_format_value(best['avg_pnl'])}."
    )


def _find_sentiment_correlation(analysis_results: AnalysisResults) -> str:
    """Summarize the daily sentiment/PnL correlation when available."""
    correlation = analysis_results.correlation_matrix
    if (
        correlation.empty
        or "sentiment_score" not in correlation.index
        or "daily_total_pnl" not in correlation.columns
    ):
        return "Daily sentiment/PnL correlation could not be computed from the available data."
    value = correlation.loc["sentiment_score", "daily_total_pnl"]
    return f"Daily sentiment score correlation with total PnL is {_format_value(value)}."


def _summarize_top_trader(advanced_results: AdvancedAnalyticsResults) -> str:
    """Summarize the highest ranked trader."""
    if advanced_results.top_traders.empty:
        return "Top trader ranking is unavailable."
    top = advanced_results.top_traders.iloc[0]
    return (
        f"Top trader by total PnL is {top['account']} with "
        f"{_format_value(top['total_pnl'])} total PnL across "
        f"{_format_value(top['total_trades'])} trades."
    )


def _recommendations() -> list[str]:
    """Provide practical recommendations for using the analysis."""
    return [
        "Compare trader performance by sentiment before increasing allocation to any strategy.",
        "Review high-leverage behavior separately during Fear and Extreme Fear regimes.",
        "Treat statistically significant results as investigation leads, not as standalone trading rules.",
        "Refresh the analysis after adding new raw data so findings reflect the latest market regimes.",
    ]


def generate_pdf_report(
    analysis_results: AnalysisResults,
    advanced_results: AdvancedAnalyticsResults,
    chart_paths: list[Path],
    output_path: Path = REPORTS_DIR / "crypto_sentiment_report.pdf",
) -> Path:
    """Generate a PDF report with methodology, charts, findings, and recommendations."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import (
            Image,
            PageBreak,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
        )
    except ImportError as error:
        raise RuntimeError(
            "reportlab is required to generate the PDF report. "
            "Install dependencies with `pip install -r requirements.txt`."
        ) from error

    output_path.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    document = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        rightMargin=0.55 * inch,
        leftMargin=0.55 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.55 * inch,
    )

    story: list[object] = [
        Paragraph("Crypto Sentiment and Trader Performance Report", styles["Title"]),
        Spacer(1, 0.2 * inch),
        Paragraph("Methodology", styles["Heading2"]),
        Paragraph(
            "The pipeline loads Fear/Greed sentiment data and Hyperliquid trader "
            "history, validates schemas, normalizes dates to UTC calendar days, "
            "cleans trader fields, engineers trade and account features, merges "
            "sentiment with performance, and evaluates profitability, leverage, "
            "symbol behavior, trader rankings, clusters, and statistical tests.",
            styles["BodyText"],
        ),
        Spacer(1, 0.2 * inch),
        Paragraph("Key Findings", styles["Heading2"]),
        Paragraph(_find_best_sentiment(analysis_results), styles["BodyText"]),
        Paragraph(_find_sentiment_correlation(analysis_results), styles["BodyText"]),
        Paragraph(_summarize_top_trader(advanced_results), styles["BodyText"]),
        Spacer(1, 0.2 * inch),
        Paragraph("Sentiment Profitability", styles["Heading2"]),
        _build_report_table(_table_data(analysis_results.sentiment_profitability)),
        Spacer(1, 0.2 * inch),
        Paragraph("Top Traders", styles["Heading2"]),
        _build_report_table(_table_data(advanced_results.top_traders)),
        Spacer(1, 0.2 * inch),
        Paragraph("Statistical Tests", styles["Heading2"]),
        _build_report_table(_table_data(advanced_results.statistical_tests)),
        Spacer(1, 0.2 * inch),
        Paragraph("Recommendations", styles["Heading2"]),
    ]

    for recommendation in _recommendations():
        story.append(Paragraph(f"- {recommendation}", styles["BodyText"]))

    existing_charts = [path for path in chart_paths if path.exists()]
    if existing_charts:
        story.extend([PageBreak(), Paragraph("Charts", styles["Heading2"])])
        for chart_path in existing_charts:
            story.append(Paragraph(chart_path.stem.replace("_", " ").title(), styles["Heading3"]))
            story.append(Image(str(chart_path), width=6.8 * inch, height=3.9 * inch))
            story.append(Spacer(1, 0.15 * inch))

    document.build(story)
    LOGGER.info("Generated PDF report: %s", output_path)
    return output_path


def _build_report_table(data: list[list[str]]) -> object:
    """Create a styled ReportLab table from row data."""
    try:
        from reportlab.lib import colors
        from reportlab.platypus import Table, TableStyle
    except ImportError as error:
        raise RuntimeError("reportlab is required to build report tables") from error

    table = Table(data, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2f3e46")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f6f8fa")]),
            ]
        )
    )
    return table
