# crypto-sentiment-analysis

End-to-end data science assignment analyzing the relationship between Bitcoin market sentiment and Hyperliquid trader performance.

## Project Scope

This project will combine:

- Fear/Greed market sentiment data from `data/raw/fear_greed_index.csv`
- Hyperliquid trader execution data from `data/raw/historical_data.csv`

The final workflow will load, validate, preprocess, merge, analyze, visualize, and report on how sentiment regimes relate to trader behavior and profitability.

## Current Status

Step 1 is complete: the repository structure, dependency file, ignore rules, and runnable entry point have been initialized.

## Expected Data Placement

Place the provided source files here:

```text
data/raw/historical_data.csv
data/raw/fear_greed_index.csv
```

Raw data files are intentionally ignored by git.

## Run

```bash
python main.py
```
