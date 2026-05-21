"""Tests for preprocessing routines."""

import pandas as pd

from src.preprocessing import preprocess_datasets


def test_preprocess_datasets_normalizes_values_and_removes_duplicates() -> None:
    """Preprocessing should clean labels, sides, duplicates, and numeric fields."""
    fear_greed = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01", "not-a-date"],
            "classification": ["extreme fear", "Greed", ""],
        }
    )
    trader_history = pd.DataFrame(
        {
            "time": [1704067200000, 1704067200000, None],
            "account": ["acct_1", "acct_1", ""],
            "symbol": ["BTC", "BTC", None],
            "closed_pnl": ["12.0", "12.0", None],
            "side": ["LONG", "LONG", "?"],
            "size": ["0.5", "0.5", None],
            "leverage": ["10x", "10x", None],
        }
    )

    cleaned_fear_greed, cleaned_trader_history, reports = preprocess_datasets(
        fear_greed,
        trader_history,
    )

    assert cleaned_fear_greed.shape[0] == 1
    assert cleaned_fear_greed.loc[0, "classification"] == "Greed"
    assert cleaned_trader_history.shape[0] == 1
    assert cleaned_trader_history.loc[0, "side"] == "buy"
    assert cleaned_trader_history.loc[0, "leverage"] == 10.0
    assert [(report.rows_dropped, report.duplicates_removed) for report in reports] == [
        (1, 1),
        (1, 1),
    ]
