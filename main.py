"""Command-line entry point for the crypto sentiment analysis project."""

from __future__ import annotations

import logging

from src.data_loader import SchemaValidationError, load_project_datasets
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
    except (FileNotFoundError, SchemaValidationError) as error:
        LOGGER.error("Dataset loading failed: %s", error)
        raise SystemExit(1) from error

    print()
    print("Loaded datasets:")
    print(f"- {fear_greed.name}: {fear_greed.dataframe.shape}")
    print(f"- {trader_history.name}: {trader_history.dataframe.shape}")


if __name__ == "__main__":
    main()
