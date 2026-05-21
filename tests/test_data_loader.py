"""Tests for robust dataset loading."""

from pathlib import Path

from src.data_loader import FEAR_GREED_SCHEMA, TRADER_SCHEMA, load_csv


FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_load_fear_greed_csv_parses_dates_and_columns() -> None:
    """Fear/Greed loader canonicalizes columns and parses dates."""
    loaded = load_csv(FIXTURES_DIR / "fear_greed_sample.csv", FEAR_GREED_SCHEMA)

    assert loaded.name == "fear_greed"
    assert loaded.dataframe.shape == (2, 2)
    assert list(loaded.dataframe.columns) == ["date", "classification"]
    assert str(loaded.dataframe["date"].dt.tz) == "UTC"


def test_load_trader_csv_skips_malformed_rows() -> None:
    """Trader loader skips malformed rows and maps common column aliases."""
    loaded = load_csv(FIXTURES_DIR / "trader_history_sample.csv", TRADER_SCHEMA)

    assert loaded.name == "trader_history"
    assert loaded.malformed_rows == 1
    assert loaded.dataframe.shape[0] == 2
    assert {"closed_pnl", "time", "leverage"}.issubset(loaded.dataframe.columns)
    assert loaded.dataframe["closed_pnl"].tolist() == [10.5, -4.0]
