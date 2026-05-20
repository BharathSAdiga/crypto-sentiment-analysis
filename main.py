"""Command-line entry point for the crypto sentiment analysis project."""

from __future__ import annotations

import logging

from src.analysis import run_exploratory_analysis, save_analysis_tables
from src.data_loader import SchemaValidationError, load_project_datasets
from src.feature_engineering import (
    engineer_features,
    merge_daily_performance_with_sentiment,
    merge_trades_with_sentiment,
)
from src.preprocessing import preprocess_datasets
from src.utils import (
    CHARTS_DIR,
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
    RAW_DATA_DIR,
    REPORTS_DIR,
    configure_logging,
    ensure_directories,
)


LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Validate project setup and load raw datasets when available."""
    configure_logging()
    ensure_directories()

    expected_files = [
        RAW_DATA_DIR / "historical_data.csv",
        RAW_DATA_DIR / "fear_greed_index.csv",
    ]

    print("crypto-sentiment-analysis")
    print("Project scaffold is ready.")
    print()
    print("Expected raw data files:")
    for file_path in expected_files:
        status = "found" if file_path.exists() else "missing"
        print(f"- {file_path.relative_to(PROJECT_ROOT)} [{status}]")
    print()
    print(f"Processed data directory: {PROCESSED_DATA_DIR.relative_to(PROJECT_ROOT)}")
    print(f"Charts directory: {CHARTS_DIR.relative_to(PROJECT_ROOT)}")
    print(f"Reports directory: {REPORTS_DIR.relative_to(PROJECT_ROOT)}")

    missing_files = [file_path for file_path in expected_files if not file_path.exists()]
    if missing_files:
        LOGGER.warning("Raw datasets are not available yet; skipping data loading.")
        return

    try:
        fear_greed, trader_history = load_project_datasets()
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
    except (FileNotFoundError, SchemaValidationError) as error:
        LOGGER.error("Dataset loading failed: %s", error)
        raise SystemExit(1) from error
    except ValueError as error:
        LOGGER.error("Dataset preprocessing failed: %s", error)
        raise SystemExit(1) from error

    print()
    print("Loaded datasets:")
    print(f"- {fear_greed.name}: {fear_greed.dataframe.shape}")
    print(f"- {trader_history.name}: {trader_history.dataframe.shape}")
    print()
    print("Preprocessed datasets:")
    print(f"- fear_greed: {cleaned_fear_greed.shape}")
    print(f"- trader_history: {cleaned_trader_history.shape}")
    for report in preprocessing_reports:
        print(
            f"- {report.dataset_name}: "
            f"dropped={report.rows_dropped}, duplicates={report.duplicates_removed}"
        )
    print()
    print("Feature tables:")
    print(f"- sentiment_features: {sentiment_features.shape}")
    print(f"- trade_features: {trade_features.shape}")
    print(f"- trader_metrics: {trader_metrics.shape}")
    print()
    print("Merged datasets:")
    print(f"- trade_sentiment: {trade_sentiment.shape}")
    print(f"- daily_sentiment: {daily_sentiment.shape}")

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    sentiment_features.to_csv(PROCESSED_DATA_DIR / "sentiment_features.csv", index=False)
    trade_features.to_csv(PROCESSED_DATA_DIR / "trade_features.csv", index=False)
    trader_metrics.to_csv(PROCESSED_DATA_DIR / "trader_metrics.csv", index=False)
    trade_sentiment.to_csv(PROCESSED_DATA_DIR / "trade_sentiment.csv", index=False)
    daily_sentiment.to_csv(PROCESSED_DATA_DIR / "daily_sentiment.csv", index=False)
    save_analysis_tables(analysis_results, REPORTS_DIR)
    print()
    print(f"Processed files saved to {PROCESSED_DATA_DIR.relative_to(PROJECT_ROOT)}")
    print(f"Analysis tables saved to {REPORTS_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
