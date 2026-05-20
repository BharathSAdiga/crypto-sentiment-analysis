"""Robust CSV loading and schema validation for project datasets."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

from src.utils import RAW_DATA_DIR


LOGGER = logging.getLogger(__name__)


class SchemaValidationError(ValueError):
    """Raised when a dataset is missing required columns."""


@dataclass(frozen=True)
class DatasetSchema:
    """Column expectations and parsing rules for a dataset."""

    name: str
    required_columns: tuple[str, ...]
    date_columns: tuple[str, ...] = ()
    numeric_columns: tuple[str, ...] = ()
    recommended_columns: tuple[str, ...] = ()
    aliases: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class LoadedDataset:
    """Loaded dataset with lightweight ingestion metadata."""

    name: str
    path: Path
    dataframe: pd.DataFrame
    malformed_rows: int


DEFAULT_COLUMN_ALIASES = {
    "date": "date",
    "classification": "classification",
    "class": "classification",
    "account": "account",
    "wallet": "account",
    "trader": "account",
    "symbol": "symbol",
    "coin": "symbol",
    "asset": "symbol",
    "executionprice": "execution_price",
    "execution_price": "execution_price",
    "price": "execution_price",
    "size": "size",
    "qty": "size",
    "quantity": "size",
    "side": "side",
    "direction": "side",
    "time": "time",
    "timestamp": "time",
    "datetime": "time",
    "startposition": "start_position",
    "start_position": "start_position",
    "position": "start_position",
    "event": "event",
    "closedpnl": "closed_pnl",
    "closed_pnl": "closed_pnl",
    "pnl": "closed_pnl",
    "leverage": "leverage",
}


FEAR_GREED_SCHEMA = DatasetSchema(
    name="fear_greed",
    required_columns=("date", "classification"),
    date_columns=("date",),
)

TRADER_SCHEMA = DatasetSchema(
    name="trader_history",
    required_columns=("time",),
    recommended_columns=("account", "symbol", "side", "closed_pnl", "size"),
    date_columns=("time",),
    numeric_columns=("execution_price", "size", "start_position", "closed_pnl", "leverage"),
)


def normalize_column_name(column_name: object) -> str:
    """Normalize a raw column name into a stable snake-case key."""
    text = str(column_name).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def canonicalize_columns(
    dataframe: pd.DataFrame,
    schema: DatasetSchema,
) -> pd.DataFrame:
    """Rename columns using built-in and schema-specific aliases."""
    aliases = {**DEFAULT_COLUMN_ALIASES, **schema.aliases}
    renamed_columns: dict[str, str] = {}

    for column in dataframe.columns:
        normalized = normalize_column_name(column)
        alias_key = normalized.replace("_", "")
        canonical_name = aliases.get(normalized, aliases.get(alias_key, normalized))
        renamed_columns[column] = canonical_name

    canonicalized = dataframe.rename(columns=renamed_columns)
    if len(set(canonicalized.columns)) != len(canonicalized.columns):
        duplicate_names = canonicalized.columns[canonicalized.columns.duplicated()].tolist()
        raise SchemaValidationError(
            f"{schema.name} has duplicate canonical columns: {duplicate_names}"
        )
    return canonicalized


def validate_schema(dataframe: pd.DataFrame, schema: DatasetSchema) -> None:
    """Validate required columns and log useful optional-column warnings."""
    columns = set(dataframe.columns)
    missing_required = sorted(set(schema.required_columns) - columns)
    if missing_required:
        raise SchemaValidationError(
            f"{schema.name} is missing required columns: {missing_required}"
        )

    missing_recommended = sorted(set(schema.recommended_columns) - columns)
    if missing_recommended:
        LOGGER.warning(
            "%s is missing recommended columns used by later analysis: %s",
            schema.name,
            missing_recommended,
        )


def parse_datetime_column(series: pd.Series) -> pd.Series:
    """Parse a date/time column, including Unix seconds, milliseconds, or nanoseconds."""
    numeric_series = pd.to_numeric(series, errors="coerce")
    numeric_ratio = numeric_series.notna().mean() if len(series) else 0

    if numeric_ratio >= 0.8:
        median_value = numeric_series.dropna().abs().median()
        if pd.isna(median_value):
            return pd.to_datetime(series, errors="coerce", utc=True)
        if median_value > 1e17:
            unit = "ns"
        elif median_value > 1e14:
            unit = "us"
        elif median_value > 1e11:
            unit = "ms"
        else:
            unit = "s"
        return pd.to_datetime(numeric_series, errors="coerce", unit=unit, utc=True)

    return pd.to_datetime(series, errors="coerce", utc=True)


def parse_dates(dataframe: pd.DataFrame, schema: DatasetSchema) -> pd.DataFrame:
    """Parse configured date columns and warn about invalid values."""
    parsed = dataframe.copy()
    for column in schema.date_columns:
        if column not in parsed.columns:
            continue
        parsed[column] = parse_datetime_column(parsed[column])
        invalid_count = int(parsed[column].isna().sum())
        if invalid_count:
            LOGGER.warning(
                "%s has %s rows with invalid %s values",
                schema.name,
                invalid_count,
                column,
            )
    return parsed


def coerce_numeric_columns(dataframe: pd.DataFrame, schema: DatasetSchema) -> pd.DataFrame:
    """Convert configured numeric columns when they are present."""
    converted = dataframe.copy()
    for column in schema.numeric_columns:
        if column in converted.columns:
            converted[column] = pd.to_numeric(converted[column], errors="coerce")
    return converted


def load_csv(path: Path, schema: DatasetSchema) -> LoadedDataset:
    """Load one CSV file with malformed-row handling and schema validation."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    malformed_rows: list[list[str]] = []

    def handle_bad_line(bad_line: list[str]) -> None:
        malformed_rows.append(bad_line)
        return None

    LOGGER.info("Loading %s from %s", schema.name, path)
    dataframe = pd.read_csv(
        path,
        engine="python",
        on_bad_lines=handle_bad_line,
        skip_blank_lines=True,
    )

    if malformed_rows:
        LOGGER.warning(
            "%s skipped %s malformed rows while reading %s",
            schema.name,
            len(malformed_rows),
            path.name,
        )

    dataframe = canonicalize_columns(dataframe, schema)
    validate_schema(dataframe, schema)
    dataframe = parse_dates(dataframe, schema)
    dataframe = coerce_numeric_columns(dataframe, schema)

    LOGGER.info(
        "Loaded %s with %s rows and %s columns",
        schema.name,
        len(dataframe),
        len(dataframe.columns),
    )
    return LoadedDataset(
        name=schema.name,
        path=path,
        dataframe=dataframe,
        malformed_rows=len(malformed_rows),
    )


def load_project_datasets(
    raw_data_dir: Path = RAW_DATA_DIR,
    loader: Callable[[Path, DatasetSchema], LoadedDataset] = load_csv,
) -> tuple[LoadedDataset, LoadedDataset]:
    """Load the Fear/Greed and trader-history datasets from the raw data directory."""
    fear_greed = loader(raw_data_dir / "fear_greed_index.csv", FEAR_GREED_SCHEMA)
    trader_history = loader(raw_data_dir / "historical_data.csv", TRADER_SCHEMA)
    return fear_greed, trader_history
