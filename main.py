"""Command-line entry point for the crypto sentiment analysis project."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.analysis import (
    run_advanced_analytics,
    run_exploratory_analysis,
    save_advanced_analytics_tables,
    save_analysis_tables,
)
from src.data_loader import SchemaValidationError, load_project_datasets
from src.feature_engineering import (
    engineer_features,
    merge_daily_performance_with_sentiment,
    merge_trades_with_sentiment,
)
from src.preprocessing import preprocess_datasets
from src.reporting import generate_pdf_report
from src.utils import (
    CHARTS_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    REPORTS_DIR,
    configure_logging,
    relative_to_project,
)
from src.visualization import generate_visualizations


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineOutputs:
    """Data and artifact outputs from a successful pipeline run."""

    fear_greed_shape: tuple[int, int]
    trader_history_shape: tuple[int, int]
    cleaned_fear_greed_shape: tuple[int, int]
    cleaned_trader_history_shape: tuple[int, int]
    sentiment_features_shape: tuple[int, int]
    trade_features_shape: tuple[int, int]
    trader_metrics_shape: tuple[int, int]
    trade_sentiment_shape: tuple[int, int]
    daily_sentiment_shape: tuple[int, int]
    preprocessing_summary: list[str]
    chart_count: int
    advanced_table_count: int
    report_path: Path | None


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the analysis pipeline."""
    parser = argparse.ArgumentParser(
        description=(
            "Analyze Bitcoin Fear/Greed sentiment against Hyperliquid trader "
            "performance."
        )
    )
    parser.add_argument(
        "--raw-data-dir",
        type=Path,
        default=RAW_DATA_DIR,
        help="Directory containing historical_data.csv and fear_greed_index.csv.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=PROCESSED_DATA_DIR,
        help="Directory for processed CSV outputs.",
    )
    parser.add_argument(
        "--charts-dir",
        type=Path,
        default=CHARTS_DIR,
        help="Directory for generated chart images.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=REPORTS_DIR,
        help="Directory for analysis tables and PDF report.",
    )
    parser.add_argument(
        "--skip-charts",
        action="store_true",
        help="Run the analysis without generating chart image files.",
    )
    parser.add_argument(
        "--skip-report",
        action="store_true",
        help="Run the analysis without generating the PDF report.",
    )
    return parser.parse_args()


def expected_raw_files(raw_data_dir: Path) -> list[Path]:
    """Return the required raw dataset paths."""
    return [
        raw_data_dir / "historical_data.csv",
        raw_data_dir / "fear_greed_index.csv",
    ]


def print_project_setup(args: argparse.Namespace) -> list[Path]:
    """Print configured paths and return missing raw files."""
    raw_files = expected_raw_files(args.raw_data_dir)

    print("crypto-sentiment-analysis")
    print("Pipeline is ready.")
    print()
    print("Expected raw data files:")
    for file_path in raw_files:
        status = "found" if file_path.exists() else "missing"
        print(f"- {relative_to_project(file_path)} [{status}]")
    print()
    print(f"Processed data directory: {relative_to_project(args.processed_dir)}")
    print(f"Charts directory: {relative_to_project(args.charts_dir)}")
    print(f"Reports directory: {relative_to_project(args.reports_dir)}")
    return [file_path for file_path in raw_files if not file_path.exists()]


def save_processed_outputs(
    processed_dir: Path,
    sentiment_features: pd.DataFrame,
    trade_features: pd.DataFrame,
    trader_metrics: pd.DataFrame,
    trade_sentiment: pd.DataFrame,
    daily_sentiment: pd.DataFrame,
) -> None:
    """Save reusable processed datasets as CSV files."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    sentiment_features.to_csv(processed_dir / "sentiment_features.csv", index=False)
    trade_features.to_csv(processed_dir / "trade_features.csv", index=False)
    trader_metrics.to_csv(processed_dir / "trader_metrics.csv", index=False)
    trade_sentiment.to_csv(processed_dir / "trade_sentiment.csv", index=False)
    daily_sentiment.to_csv(processed_dir / "daily_sentiment.csv", index=False)


def run_pipeline(args: argparse.Namespace) -> PipelineOutputs | None:
    """Run the complete analysis pipeline and return output metadata."""
    missing_files = print_project_setup(args)
    if missing_files:
        LOGGER.warning("Raw datasets are not available yet; skipping data loading.")
        return None

    try:
        fear_greed, trader_history = load_project_datasets(raw_data_dir=args.raw_data_dir)
        cleaned_fear_greed, cleaned_trader_history, preprocessing_reports = (
            preprocess_datasets(fear_greed.dataframe, trader_history.dataframe)
        )
        sentiment_features, trade_features, trader_metrics = engineer_features(
            cleaned_fear_greed,
            cleaned_trader_history,
        )
        trade_sentiment = merge_trades_with_sentiment(
            trade_features,
            sentiment_features,
        )
        daily_sentiment = merge_daily_performance_with_sentiment(
            trade_features,
            sentiment_features,
        )
        analysis_results = run_exploratory_analysis(
            trade_sentiment,
            daily_sentiment,
            trader_metrics,
        )
        advanced_results = run_advanced_analytics(
            trade_sentiment,
            trader_metrics,
        )
    except (FileNotFoundError, SchemaValidationError) as error:
        LOGGER.error("Dataset loading failed: %s", error)
        raise SystemExit(1) from error
    except ValueError as error:
        LOGGER.error("Dataset processing failed: %s", error)
        raise SystemExit(1) from error

    chart_paths = []
    if not args.skip_charts:
        chart_paths = generate_visualizations(
            trade_sentiment,
            daily_sentiment,
            trader_metrics,
            analysis_results,
            args.charts_dir,
        )

    report_path = None
    if not args.skip_report:
        try:
            report_path = generate_pdf_report(
                analysis_results,
                advanced_results,
                chart_paths,
                args.reports_dir / "crypto_sentiment_report.pdf",
            )
        except RuntimeError as error:
            LOGGER.error("Report generation failed: %s", error)
            raise SystemExit(1) from error

    save_processed_outputs(
        args.processed_dir,
        sentiment_features,
        trade_features,
        trader_metrics,
        trade_sentiment,
        daily_sentiment,
    )
    save_analysis_tables(analysis_results, args.reports_dir)
    save_advanced_analytics_tables(advanced_results, args.reports_dir)

    return PipelineOutputs(
        fear_greed_shape=fear_greed.dataframe.shape,
        trader_history_shape=trader_history.dataframe.shape,
        cleaned_fear_greed_shape=cleaned_fear_greed.shape,
        cleaned_trader_history_shape=cleaned_trader_history.shape,
        sentiment_features_shape=sentiment_features.shape,
        trade_features_shape=trade_features.shape,
        trader_metrics_shape=trader_metrics.shape,
        trade_sentiment_shape=trade_sentiment.shape,
        daily_sentiment_shape=daily_sentiment.shape,
        preprocessing_summary=[
            f"- {report.dataset_name}: "
            f"dropped={report.rows_dropped}, duplicates={report.duplicates_removed}"
            for report in preprocessing_reports
        ],
        chart_count=len(chart_paths),
        advanced_table_count=len(advanced_results.__dict__),
        report_path=report_path,
    )


def print_outputs(outputs: PipelineOutputs, args: argparse.Namespace) -> None:
    """Print a concise run summary."""
    print()
    print("Loaded datasets:")
    print(f"- fear_greed: {outputs.fear_greed_shape}")
    print(f"- trader_history: {outputs.trader_history_shape}")
    print()
    print("Preprocessed datasets:")
    print(f"- fear_greed: {outputs.cleaned_fear_greed_shape}")
    print(f"- trader_history: {outputs.cleaned_trader_history_shape}")
    for summary in outputs.preprocessing_summary:
        print(summary)
    print()
    print("Feature tables:")
    print(f"- sentiment_features: {outputs.sentiment_features_shape}")
    print(f"- trade_features: {outputs.trade_features_shape}")
    print(f"- trader_metrics: {outputs.trader_metrics_shape}")
    print()
    print("Merged datasets:")
    print(f"- trade_sentiment: {outputs.trade_sentiment_shape}")
    print(f"- daily_sentiment: {outputs.daily_sentiment_shape}")
    print()
    print(f"Processed files saved to {relative_to_project(args.processed_dir)}")
    print(f"Analysis tables saved to {relative_to_project(args.reports_dir)}")
    print(f"Charts saved: {outputs.chart_count}")
    print(f"Advanced analytics tables saved: {outputs.advanced_table_count}")
    if outputs.report_path is not None:
        print(f"PDF report saved to {relative_to_project(outputs.report_path)}")


def main() -> None:
    """Run the command-line analysis workflow."""
    configure_logging()
    args = parse_args()
    outputs = run_pipeline(args)
    if outputs is not None:
        print_outputs(outputs, args)


if __name__ == "__main__":
    main()
