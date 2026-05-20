"""Feature engineering for sentiment and trader-performance analysis."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)


SENTIMENT_SCORE_MAP = {
    "Extreme Fear": 1,
    "Fear": 2,
    "Neutral": 3,
    "Greed": 4,
    "Extreme Greed": 5,
}

LEVERAGE_BINS = [-np.inf, 1, 3, 5, 10, 20, np.inf]
LEVERAGE_LABELS = [
    "1x or less",
    "1x-3x",
    "3x-5x",
    "5x-10x",
    "10x-20x",
    "20x+",
]


def add_sentiment_features(fear_greed: pd.DataFrame) -> pd.DataFrame:
    """Add ordinal sentiment scores to the cleaned Fear/Greed dataset."""
    if "classification" not in fear_greed.columns:
        raise ValueError("Fear/Greed data must include classification")

    featured = fear_greed.copy()
    featured["sentiment_score"] = featured["classification"].map(SENTIMENT_SCORE_MAP)
    featured["is_extreme_sentiment"] = featured["classification"].isin(
        ("Extreme Fear", "Extreme Greed")
    )

    missing_scores = int(featured["sentiment_score"].isna().sum())
    if missing_scores:
        LOGGER.warning("Sentiment score missing for %s rows", missing_scores)

    return featured


def add_trade_features(trader_history: pd.DataFrame) -> pd.DataFrame:
    """Add row-level trade metrics used by later analysis."""
    required_columns = {"closed_pnl", "side", "size", "leverage"}
    missing_columns = sorted(required_columns - set(trader_history.columns))
    if missing_columns:
        raise ValueError(f"Trader history data is missing columns: {missing_columns}")

    featured = trader_history.copy()
    featured["pnl_per_trade"] = featured["closed_pnl"]
    featured["is_win"] = featured["closed_pnl"] > 0
    featured["trade_direction"] = featured["side"].where(
        featured["side"].isin(("buy", "sell")),
        "unknown",
    )
    featured["leverage_bucket"] = pd.cut(
        featured["leverage"],
        bins=LEVERAGE_BINS,
        labels=LEVERAGE_LABELS,
    ).astype("object")
    featured["leverage_bucket"] = featured["leverage_bucket"].fillna("unknown")
    featured["pnl_per_size"] = np.where(
        featured["size"].abs() > 0,
        featured["closed_pnl"] / featured["size"].abs(),
        np.nan,
    )

    if "execution_price" in featured.columns:
        featured["trade_notional"] = (
            featured["execution_price"].abs() * featured["size"].abs()
        )
    else:
        featured["trade_notional"] = np.nan

    return featured


def _profit_factor(closed_pnl: pd.Series) -> float:
    """Calculate profit factor as gross profit divided by absolute gross loss."""
    gross_profit = closed_pnl[closed_pnl > 0].sum()
    gross_loss = closed_pnl[closed_pnl < 0].sum()
    if gross_loss < 0:
        return float(gross_profit / abs(gross_loss))
    if gross_profit > 0:
        return float("inf")
    return 0.0


def build_trader_metrics(trade_features: pd.DataFrame) -> pd.DataFrame:
    """Aggregate trader-level performance and behavior metrics."""
    if "account" not in trade_features.columns:
        raise ValueError("Trade features must include account")

    grouped = trade_features.groupby("account", dropna=False)
    metrics = grouped.agg(
        total_trades=("closed_pnl", "size"),
        win_rate=("is_win", "mean"),
        avg_pnl=("closed_pnl", "mean"),
        total_pnl=("closed_pnl", "sum"),
        avg_leverage=("leverage", "mean"),
        buy_trades=("trade_direction", lambda values: int((values == "buy").sum())),
        sell_trades=("trade_direction", lambda values: int((values == "sell").sum())),
        avg_size=("size", "mean"),
        pnl_volatility=("closed_pnl", lambda values: float(values.std(ddof=0))),
        avg_pnl_per_size=("pnl_per_size", "mean"),
    ).reset_index()

    metrics["buy_sell_ratio"] = np.where(
        metrics["sell_trades"] > 0,
        metrics["buy_trades"] / metrics["sell_trades"],
        np.where(metrics["buy_trades"] > 0, np.inf, 0.0),
    )

    profit_factors = grouped["closed_pnl"].apply(_profit_factor).reset_index()
    profit_factors = profit_factors.rename(columns={"closed_pnl": "profit_factor"})
    metrics = metrics.merge(profit_factors, on="account", how="left")

    metrics = metrics.sort_values(
        ["total_pnl", "win_rate", "total_trades"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    return metrics


def normalize_utc_date(series: pd.Series) -> pd.Series:
    """Convert datetimes to UTC-normalized calendar dates."""
    return pd.to_datetime(series, errors="coerce", utc=True).dt.normalize()


def aggregate_daily_trader_performance(trade_features: pd.DataFrame) -> pd.DataFrame:
    """Aggregate row-level trades into daily performance metrics."""
    if "time" not in trade_features.columns:
        raise ValueError("Trade features must include time")

    daily = trade_features.copy()
    daily["date"] = normalize_utc_date(daily["time"])
    daily = daily.dropna(subset=["date"])

    grouped = daily.groupby("date", dropna=False)
    aggregated = grouped.agg(
        daily_trades=("closed_pnl", "size"),
        active_traders=("account", "nunique"),
        symbols_traded=("symbol", "nunique"),
        daily_total_pnl=("closed_pnl", "sum"),
        daily_avg_pnl=("closed_pnl", "mean"),
        daily_win_rate=("is_win", "mean"),
        daily_avg_leverage=("leverage", "mean"),
        daily_avg_size=("size", "mean"),
        daily_pnl_volatility=("closed_pnl", lambda values: float(values.std(ddof=0))),
        daily_buy_trades=("trade_direction", lambda values: int((values == "buy").sum())),
        daily_sell_trades=("trade_direction", lambda values: int((values == "sell").sum())),
    ).reset_index()

    aggregated["daily_buy_sell_ratio"] = np.where(
        aggregated["daily_sell_trades"] > 0,
        aggregated["daily_buy_trades"] / aggregated["daily_sell_trades"],
        np.where(aggregated["daily_buy_trades"] > 0, np.inf, 0.0),
    )
    return aggregated.sort_values("date").reset_index(drop=True)


def merge_trades_with_sentiment(
    trade_features: pd.DataFrame,
    sentiment_features: pd.DataFrame,
) -> pd.DataFrame:
    """Attach same-day sentiment labels and scores to each trade."""
    if "date" not in sentiment_features.columns:
        raise ValueError("Sentiment features must include date")

    trades = trade_features.copy()
    sentiment = sentiment_features.copy()
    trades["date"] = normalize_utc_date(trades["time"])
    sentiment["date"] = normalize_utc_date(sentiment["date"])

    merged = trades.merge(
        sentiment[["date", "classification", "sentiment_score", "is_extreme_sentiment"]],
        on="date",
        how="left",
    )
    return merged.sort_values(["date", "time"]).reset_index(drop=True)


def merge_daily_performance_with_sentiment(
    trade_features: pd.DataFrame,
    sentiment_features: pd.DataFrame,
) -> pd.DataFrame:
    """Merge daily trader-performance aggregates with same-day sentiment."""
    daily_performance = aggregate_daily_trader_performance(trade_features)
    sentiment = sentiment_features.copy()
    sentiment["date"] = normalize_utc_date(sentiment["date"])

    merged = daily_performance.merge(
        sentiment[["date", "classification", "sentiment_score", "is_extreme_sentiment"]],
        on="date",
        how="left",
    )
    return merged.sort_values("date").reset_index(drop=True)


def engineer_features(
    fear_greed: pd.DataFrame,
    trader_history: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create sentiment, trade, and trader-level feature tables."""
    sentiment_features = add_sentiment_features(fear_greed)
    trade_features = add_trade_features(trader_history)
    trader_metrics = build_trader_metrics(trade_features)
    return sentiment_features, trade_features, trader_metrics
