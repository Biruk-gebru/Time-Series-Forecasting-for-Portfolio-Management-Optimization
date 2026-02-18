"""Unit tests for Task 1 – data extraction and cleaning."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest


DATA_DIR = Path(__file__).parents[1] / "data" / "processed"
TICKERS = ["TSLA", "BND", "SPY"]


def _read_clean_csv(ticker: str) -> pd.DataFrame:
    """Read a clean CSV, skipping the yfinance MultiIndex 'Ticker' header row."""
    path = DATA_DIR / f"{ticker}_clean.csv"
    # yfinance saves a MultiIndex header: row 0 = Price names, row 1 = Ticker name.
    # When read back, the first data row is the Ticker label row which becomes NaN.
    # We use header=[0,1] to read both header rows, then flatten.
    df = pd.read_csv(path, header=[0, 1], index_col=0)
    # Drop the ticker-level from the MultiIndex columns, keep just the price names
    df.columns = df.columns.get_level_values(0)
    # Drop any fully-NaN rows (the residual Ticker label row)
    df = df.dropna(how="all")
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[df.index.notna()]
    return df


@pytest.mark.parametrize("ticker", TICKERS)
def test_clean_csv_exists(ticker: str) -> None:
    """Cleaned CSV must exist after running the notebook."""
    path = DATA_DIR / f"{ticker}_clean.csv"
    assert path.exists(), f"Missing: {path}"


@pytest.mark.parametrize("ticker", TICKERS)
def test_clean_csv_no_nulls(ticker: str) -> None:
    """Cleaned CSV must have zero null values."""
    path = DATA_DIR / f"{ticker}_clean.csv"
    if not path.exists():
        pytest.skip("CSV not yet generated — run task1_eda.ipynb first")
    df = _read_clean_csv(ticker)
    assert df.isnull().sum().sum() == 0, f"{ticker} has NaN values"


@pytest.mark.parametrize("ticker", TICKERS)
def test_clean_csv_has_required_columns(ticker: str) -> None:
    """Cleaned CSV must contain OHLCV columns."""
    path = DATA_DIR / f"{ticker}_clean.csv"
    if not path.exists():
        pytest.skip("CSV not yet generated — run task1_eda.ipynb first")
    df = _read_clean_csv(ticker)
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        assert col in df.columns, f"{ticker} missing column: {col}"


def test_stats_json_exists() -> None:
    """task1_stats.json must exist after running the notebook."""
    path = DATA_DIR / "task1_stats.json"
    assert path.exists(), f"Missing: {path}"


def test_stats_json_has_required_keys() -> None:
    """Stats JSON must contain expected top-level keys."""
    path = DATA_DIR / "task1_stats.json"
    if not path.exists():
        pytest.skip("Stats JSON not yet generated — run task1_eda.ipynb first")
    with open(path) as f:
        stats = json.load(f)
    for key in ["adf_results", "risk_metrics", "return_stats", "row_counts"]:
        assert key in stats, f"Missing key in stats JSON: {key}"


def test_images_exist() -> None:
    """Key figures must be saved after running the notebook."""
    img_dir = Path(__file__).parents[1] / "notebooks" / "images"
    expected = [
        "fig1_closing_prices.png",
        "fig2_daily_returns.png",
        "fig3_tsla_rolling_stats.png",
        "fig4_return_distributions.png",
        "fig5_correlation_heatmap.png",
        "fig6_tsla_var.png",
    ]
    for fname in expected:
        path = img_dir / fname
        assert path.exists(), f"Missing figure: {path}"
