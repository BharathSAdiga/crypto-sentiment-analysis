"""Tests for feature engineering outputs."""

import numpy as np
import pandas as pd

from src.feature_engineering import (
    engineer_features,
    merge_daily_performance_with_sentiment,
    merge_trades_with_sentiment,
)
from src.preprocessing import preprocess_datasets


def _prepared_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    fear_greed = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02"],
            "classification": ["Extreme Fear", "Greed"],
        }
    )
    trader_history = pd.DataFrame(
        {
            "time": [
                "2024-01-01T03:00:00Z",
                "2024-01-01T23:00:00Z",
                "2024-01-02T01:00:00Z",
            ],
            "account": ["acct_1", "acct_1", "acct_2"],
            "symbol": ["BTC", "BTC", "ETH"],
            "closed_pnl": [10.0, -5.0, 8.0],
            "side": ["buy", "sell", "buy"],
            "size": [1.0, 2.0, 1.0],
            "leverage": [2.0, 12.0, 25.0],
            "execution_price": [40000.0, 41000.0, 2000.0],
        }
    )
    cleaned_fear_greed, cleaned_trader_history, _ = preprocess_datasets(
        fear_greed,
        trader_history,
    )
    return cleaned_fear_greed, cleaned_trader_history


def test_engineer_features_creates_required_metrics() -> None:
    """Feature engineering should produce sentiment, trade, and trader metrics."""
    fear_greed, trader_history = _prepared_inputs()

    sentiment_features, trade_features, trader_metrics = engineer_features(
        fear_greed,
        trader_history,
    )

    assert sentiment_features["sentiment_score"].tolist() == [1, 4]
    assert trade_features["leverage_bucket"].tolist() == [
        "1x-3x",
        "10x-20x",
        "20x+",
    ]

    acct_1 = trader_metrics.loc[trader_metrics["account"] == "acct_1"].iloc[0]
    assert acct_1["total_trades"] == 2
    assert acct_1["win_rate"] == 0.5
    assert acct_1["total_pnl"] == 5.0
    assert acct_1["buy_sell_ratio"] == 1.0
    assert acct_1["profit_factor"] == 2.0

    acct_2 = trader_metrics.loc[trader_metrics["account"] == "acct_2"].iloc[0]
    assert np.isinf(acct_2["buy_sell_ratio"])


def test_merge_outputs_align_on_utc_date() -> None:
    """Trade and daily merges should align trades to same-day UTC sentiment."""
    fear_greed, trader_history = _prepared_inputs()
    sentiment_features, trade_features, _ = engineer_features(
        fear_greed,
        trader_history,
    )

    trade_sentiment = merge_trades_with_sentiment(trade_features, sentiment_features)
    daily_sentiment = merge_daily_performance_with_sentiment(
        trade_features,
        sentiment_features,
    )

    assert trade_sentiment["classification"].tolist() == [
        "Extreme Fear",
        "Extreme Fear",
        "Greed",
    ]
    assert daily_sentiment["daily_trades"].tolist() == [2, 1]
    assert daily_sentiment["daily_total_pnl"].tolist() == [5.0, 8.0]
