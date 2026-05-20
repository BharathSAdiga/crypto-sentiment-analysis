"""Command-line entry point for the crypto sentiment analysis project."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
CHARTS_DIR = PROJECT_ROOT / "outputs" / "charts"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"


def main() -> None:
    """Print the current project setup and expected input files."""
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


if __name__ == "__main__":
    main()
