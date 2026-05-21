# crypto-sentiment-analysis

End-to-end data science project analyzing how Bitcoin market sentiment, represented by the Fear/Greed Index, relates to Hyperliquid trader performance.

## What This Project Does

The pipeline:

1. Loads raw Fear/Greed and Hyperliquid trader CSV files.
2. Validates schemas and parses dates robustly.
3. Cleans nulls, duplicates, sentiment labels, trade sides, and numeric fields.
4. Engineers trader, trade, and sentiment features.
5. Aligns trades to UTC calendar dates and merges daily sentiment.
6. Runs exploratory analysis, advanced trader analytics, and statistical tests.
7. Generates reusable CSV outputs, charts, and a PDF report.

## Project Structure

```text
crypto-sentiment-analysis/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── analysis.ipynb
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── analysis.py
│   ├── visualization.py
│   ├── reporting.py
│   └── utils.py
├── outputs/
│   ├── charts/
│   └── reports/
├── tests/
├── main.py
├── requirements.txt
└── README.md
```

## Data Placement

Place the provided datasets in `data/raw/`:

```text
data/raw/historical_data.csv
data/raw/fear_greed_index.csv
```

Raw data is intentionally ignored by git.

Expected Fear/Greed columns:

- `Date`
- `Classification`

Expected Hyperliquid columns can include:

- `account`
- `symbol`
- `execution price`
- `size`
- `side`
- `time`
- `start position`
- `event`
- `closedPnL`
- `leverage`

The loader accepts common column-name variations and normalizes them internally.

## Setup

```bash
python -m pip install -r requirements.txt
```

## Run the Full Pipeline

```bash
python main.py
```

Useful CLI options:

```bash
python main.py --skip-report
python main.py --skip-charts
python main.py --raw-data-dir data/raw --processed-dir data/processed
```

If the raw CSV files are missing, the command reports the expected paths and exits cleanly.

## Outputs

Processed CSVs are written to `data/processed/`:

- `sentiment_features.csv`
- `trade_features.csv`
- `trader_metrics.csv`
- `trade_sentiment.csv`
- `daily_sentiment.csv`

Analysis tables and the PDF report are written to `outputs/reports/`.

Charts are written to `outputs/charts/`, including:

- PnL distribution
- Sentiment profitability
- Correlation heatmap
- PnL by sentiment boxplot
- Top trader comparison
- Daily PnL trend
- Leverage by sentiment

## Tests

```bash
python -B -m pytest
```

The `-B` flag prevents Python from writing bytecode files, which keeps test runs cleaner on restrictive Windows workspaces.

## Notebook

Open `notebooks/analysis.ipynb` for an exploratory workflow that reuses the production modules.
