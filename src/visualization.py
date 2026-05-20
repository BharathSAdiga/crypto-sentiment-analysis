"""Visualization pipeline for the crypto sentiment analysis project."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.analysis import AnalysisResults
from src.utils import CHARTS_DIR


LOGGER = logging.getLogger(__name__)


def _prepare_output_dir(output_dir: Path) -> Path:
    """Create the chart output directory and return it."""
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _save_current_figure(path: Path) -> Path:
    """Save and close the current Matplotlib figure."""
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    LOGGER.info("Saved chart: %s", path)
    return path


def plot_pnl_distribution(trade_sentiment: pd.DataFrame, output_dir: Path) -> Path | None:
    """Create a histogram of trade-level closed PnL."""
    if trade_sentiment.empty or "closed_pnl" not in trade_sentiment.columns:
        return None

    plt.figure(figsize=(10, 6))
    sns.histplot(trade_sentiment["closed_pnl"], bins=40, kde=True, color="#2f6f9f")
    plt.title("Trade PnL Distribution")
    plt.xlabel("Closed PnL")
    plt.ylabel("Trades")
    return _save_current_figure(output_dir / "pnl_distribution.png")


def plot_sentiment_profitability(
    analysis_results: AnalysisResults,
    output_dir: Path,
) -> Path | None:
    """Create a bar chart of total PnL by sentiment classification."""
    data = analysis_results.sentiment_profitability
    if data.empty:
        return None

    plt.figure(figsize=(10, 6))
    sns.barplot(data=data, x="classification", y="total_pnl", hue="classification")
    plt.title("Total PnL by Market Sentiment")
    plt.xlabel("Sentiment")
    plt.ylabel("Total PnL")
    plt.xticks(rotation=20, ha="right")
    plt.legend([], [], frameon=False)
    return _save_current_figure(output_dir / "sentiment_profitability.png")


def plot_correlation_heatmap(
    analysis_results: AnalysisResults,
    output_dir: Path,
) -> Path | None:
    """Create a heatmap for daily numeric correlations."""
    correlation = analysis_results.correlation_matrix
    if correlation.empty:
        return None

    plt.figure(figsize=(12, 9))
    sns.heatmap(
        correlation,
        cmap="vlag",
        center=0,
        linewidths=0.3,
        cbar_kws={"label": "Pearson correlation"},
    )
    plt.title("Daily Metric Correlations")
    return _save_current_figure(output_dir / "correlation_heatmap.png")


def plot_pnl_by_sentiment_boxplot(
    trade_sentiment: pd.DataFrame,
    output_dir: Path,
) -> Path | None:
    """Create a boxplot of trade PnL by sentiment."""
    required_columns = {"classification", "closed_pnl"}
    if trade_sentiment.empty or not required_columns.issubset(trade_sentiment.columns):
        return None

    data = trade_sentiment.copy()
    data["classification"] = data["classification"].fillna("Unknown")

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data, x="classification", y="closed_pnl", color="#75aadb")
    plt.title("Trade PnL by Sentiment")
    plt.xlabel("Sentiment")
    plt.ylabel("Closed PnL")
    plt.xticks(rotation=20, ha="right")
    return _save_current_figure(output_dir / "pnl_by_sentiment_boxplot.png")


def plot_top_trader_comparison(
    trader_metrics: pd.DataFrame,
    output_dir: Path,
    top_n: int = 10,
) -> Path | None:
    """Create a trader comparison bar chart for top accounts by total PnL."""
    required_columns = {"account", "total_pnl"}
    if trader_metrics.empty or not required_columns.issubset(trader_metrics.columns):
        return None

    data = trader_metrics.nlargest(top_n, "total_pnl").copy()
    plt.figure(figsize=(12, 6))
    sns.barplot(data=data, x="account", y="total_pnl", hue="account")
    plt.title(f"Top {len(data)} Traders by Total PnL")
    plt.xlabel("Account")
    plt.ylabel("Total PnL")
    plt.xticks(rotation=45, ha="right")
    plt.legend([], [], frameon=False)
    return _save_current_figure(output_dir / "top_trader_comparison.png")


def plot_daily_pnl_trend(daily_sentiment: pd.DataFrame, output_dir: Path) -> Path | None:
    """Create a trend chart for daily PnL and sentiment score."""
    required_columns = {"date", "daily_total_pnl", "sentiment_score"}
    if daily_sentiment.empty or not required_columns.issubset(daily_sentiment.columns):
        return None

    data = daily_sentiment.sort_values("date")
    fig, axis_pnl = plt.subplots(figsize=(12, 6))
    axis_pnl.plot(data["date"], data["daily_total_pnl"], color="#1f77b4", label="Daily PnL")
    axis_pnl.set_title("Daily PnL and Sentiment Trend")
    axis_pnl.set_xlabel("Date")
    axis_pnl.set_ylabel("Daily Total PnL")
    axis_pnl.tick_params(axis="x", rotation=30)

    axis_sentiment = axis_pnl.twinx()
    axis_sentiment.plot(
        data["date"],
        data["sentiment_score"],
        color="#c44e52",
        label="Sentiment Score",
        alpha=0.75,
    )
    axis_sentiment.set_ylabel("Sentiment Score")
    axis_sentiment.set_ylim(0.5, 5.5)
    fig.legend(loc="upper right", bbox_to_anchor=(0.92, 0.9))
    return _save_current_figure(output_dir / "daily_pnl_trend.png")


def plot_leverage_by_sentiment(
    analysis_results: AnalysisResults,
    output_dir: Path,
) -> Path | None:
    """Create a bar chart of average leverage by sentiment and leverage bucket."""
    data = analysis_results.leverage_by_sentiment
    if data.empty:
        return None

    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=data,
        x="classification",
        y="avg_leverage",
        hue="leverage_bucket",
    )
    plt.title("Average Leverage by Sentiment and Bucket")
    plt.xlabel("Sentiment")
    plt.ylabel("Average Leverage")
    plt.xticks(rotation=20, ha="right")
    plt.legend(title="Leverage Bucket", bbox_to_anchor=(1.02, 1), loc="upper left")
    return _save_current_figure(output_dir / "leverage_by_sentiment.png")


def generate_visualizations(
    trade_sentiment: pd.DataFrame,
    daily_sentiment: pd.DataFrame,
    trader_metrics: pd.DataFrame,
    analysis_results: AnalysisResults,
    output_dir: Path = CHARTS_DIR,
) -> list[Path]:
    """Generate the full visualization suite and return saved chart paths."""
    output_dir = _prepare_output_dir(output_dir)
    sns.set_theme(style="whitegrid", context="notebook")

    chart_paths = [
        plot_pnl_distribution(trade_sentiment, output_dir),
        plot_sentiment_profitability(analysis_results, output_dir),
        plot_correlation_heatmap(analysis_results, output_dir),
        plot_pnl_by_sentiment_boxplot(trade_sentiment, output_dir),
        plot_top_trader_comparison(trader_metrics, output_dir),
        plot_daily_pnl_trend(daily_sentiment, output_dir),
        plot_leverage_by_sentiment(analysis_results, output_dir),
    ]
    return [path for path in chart_paths if path is not None]
