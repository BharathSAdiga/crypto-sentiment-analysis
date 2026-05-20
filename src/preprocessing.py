"""Preprocessing routines for sentiment and trader-performance datasets."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data_loader import parse_datetime_column


LOGGER = logging.getLogger(__name__)


SENTIMENT_LABELS = {
    "extreme fear": "Extreme Fear",
    "fear": "Fear",
    "neutral": "Neutral",
    "greed": "Greed",
    "extreme greed": "Extreme Greed",
}

BUY_VALUES = {
    "buy",
    "b",
    "long",
    "open long",
    "close short",
}

SELL_VALUES = {
    "sell",
    "s",
    "short",
    "open short",
    "close long",
}

NUMERIC_TRADE_COLUMNS = (
    "execution_price",
    "size",
    "start_position",
    "closed_pnl",
    "leverage",
)


@dataclass(frozen=True)
class PreprocessingReport:
    """Basic row-level audit information for a preprocessing step."""

    dataset_name: str
    input_rows: int
    output_rows: int
    duplicates_removed: int
    rows_dropped: int


def clean_text(value: object) -> str | None:
    """Return a normalized text value or None when the input is blank."""
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text if text else None


def normalize_sentiment(value: object) -> str | None:
    """Normalize Fear/Greed classification values to canonical labels."""
    text = clean_text(value)
    if text is None:
        return None

    normalized = re.sub(r"[^a-z]+", " ", text.lower()).strip()
    return SENTIMENT_LABELS.get(normalized)


def normalize_trade_side(value: object) -> str:
    """Normalize trade direction values to buy, sell, or unknown."""
    text = clean_text(value)
    if text is None:
        return "unknown"

    normalized = re.sub(r"[^a-z]+", " ", text.lower()).strip()
    if normalized in BUY_VALUES:
        return "buy"
    if normalized in SELL_VALUES:
        return "sell"
    return "unknown"


def coerce_numeric_value(value: object) -> float:
    """Convert common numeric strings, including leverage like '10x', to float."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    text = str(value).replace(",", "").strip()
    match = re.search(r"-?\d+(\.\d+)?", text)
    if not match:
        return np.nan
    return float(match.group(0))


def ensure_columns(dataframe: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    """Ensure optional columns exist so later pipeline stages can rely on them."""
    prepared = dataframe.copy()
    for column in columns:
        if column not in prepared.columns:
            prepared[column] = np.nan
    return prepared


def preprocess_fear_greed(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, PreprocessingReport]:
    """Clean and normalize the Fear/Greed sentiment dataset."""
    required_columns = {"date", "classification"}
    missing_columns = sorted(required_columns - set(dataframe.columns))
    if missing_columns:
        raise ValueError(f"Fear/Greed data is missing columns: {missing_columns}")

    input_rows = len(dataframe)
    cleaned = dataframe.copy()
    cleaned["date"] = parse_datetime_column(cleaned["date"]).dt.normalize()
    cleaned["classification"] = cleaned["classification"].map(normalize_sentiment)

    before_drop = len(cleaned)
    cleaned = cleaned.dropna(subset=["date", "classification"])
    rows_dropped = before_drop - len(cleaned)

    before_duplicates = len(cleaned)
    cleaned = cleaned.drop_duplicates(subset=["date"], keep="last")
    duplicates_removed = before_duplicates - len(cleaned)

    cleaned = cleaned.sort_values("date").reset_index(drop=True)
    report = PreprocessingReport(
        dataset_name="fear_greed",
        input_rows=input_rows,
        output_rows=len(cleaned),
        duplicates_removed=duplicates_removed,
        rows_dropped=rows_dropped,
    )
    LOGGER.info("Preprocessed Fear/Greed data: %s", report)
    return cleaned, report


def preprocess_trader_history(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, PreprocessingReport]:
    """Clean and normalize Hyperliquid trader-history data."""
    if "time" not in dataframe.columns:
        raise ValueError("Trader history data is missing required column: time")

    input_rows = len(dataframe)
    cleaned = dataframe.copy()
    cleaned = ensure_columns(
        cleaned,
        (
            "account",
            "symbol",
            "execution_price",
            "size",
            "side",
            "start_position",
            "event",
            "closed_pnl",
            "leverage",
        ),
    )

    cleaned["time"] = parse_datetime_column(cleaned["time"])
    cleaned["account"] = cleaned["account"].map(clean_text).fillna("unknown_account")
    cleaned["symbol"] = cleaned["symbol"].map(clean_text).fillna("UNKNOWN")
    cleaned["event"] = cleaned["event"].map(clean_text).fillna("unknown")
    cleaned["side"] = cleaned["side"].map(normalize_trade_side)

    for column in NUMERIC_TRADE_COLUMNS:
        cleaned[column] = cleaned[column].map(coerce_numeric_value)

    cleaned["closed_pnl"] = cleaned["closed_pnl"].fillna(0.0)
    cleaned["size"] = cleaned["size"].fillna(0.0)

    before_drop = len(cleaned)
    cleaned = cleaned.dropna(subset=["time"])
    rows_dropped = before_drop - len(cleaned)

    before_duplicates = len(cleaned)
    cleaned = cleaned.drop_duplicates()
    duplicates_removed = before_duplicates - len(cleaned)

    cleaned = cleaned.sort_values("time").reset_index(drop=True)
    report = PreprocessingReport(
        dataset_name="trader_history",
        input_rows=input_rows,
        output_rows=len(cleaned),
        duplicates_removed=duplicates_removed,
        rows_dropped=rows_dropped,
    )
    LOGGER.info("Preprocessed trader history data: %s", report)
    return cleaned, report


def preprocess_datasets(
    fear_greed: pd.DataFrame,
    trader_history: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[PreprocessingReport]]:
    """Preprocess both project datasets and return cleaned frames with reports."""
    cleaned_fear_greed, fear_greed_report = preprocess_fear_greed(fear_greed)
    cleaned_trader_history, trader_report = preprocess_trader_history(trader_history)
    return cleaned_fear_greed, cleaned_trader_history, [fear_greed_report, trader_report]
