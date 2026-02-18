"""Unit tests for Task 2 – forecasting model outputs."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

DATA_DIR = Path(__file__).parents[1] / "data" / "processed"
IMG_DIR  = Path(__file__).parents[1] / "notebooks" / "images"


# ── Forecast CSV tests ────────────────────────────────────────────────────────

@pytest.mark.parametrize("fname", ["arima_forecast.csv", "lstm_forecast.csv"])
def test_forecast_csv_exists(fname: str) -> None:
    """Forecast CSVs must exist after running the notebook."""
    assert (DATA_DIR / fname).exists(), f"Missing: {fname}"


@pytest.mark.parametrize("fname,col", [
    ("arima_forecast.csv", "arima_forecast"),
    ("lstm_forecast.csv",  "lstm_forecast"),
])
def test_forecast_csv_no_nulls(fname: str, col: str) -> None:
    """Forecast CSVs must have no null values."""
    path = DATA_DIR / fname
    if not path.exists():
        pytest.skip(f"{fname} not yet generated — run task2_forecasting.ipynb first")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    assert df.isnull().sum().sum() == 0, f"{fname} has NaN values"


@pytest.mark.parametrize("fname,col", [
    ("arima_forecast.csv", "arima_forecast"),
    ("lstm_forecast.csv",  "lstm_forecast"),
])
def test_forecast_csv_positive_prices(fname: str, col: str) -> None:
    """All forecast prices must be positive."""
    path = DATA_DIR / fname
    if not path.exists():
        pytest.skip(f"{fname} not yet generated — run task2_forecasting.ipynb first")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    assert (df[col] > 0).all(), f"{fname} contains non-positive price values"


# ── Stats JSON tests ──────────────────────────────────────────────────────────

def test_task2_stats_exists() -> None:
    """task2_stats.json must exist after running the notebook."""
    assert (DATA_DIR / "task2_stats.json").exists(), "Missing task2_stats.json"


def test_task2_stats_has_required_keys() -> None:
    """Stats JSON must contain expected keys."""
    path = DATA_DIR / "task2_stats.json"
    if not path.exists():
        pytest.skip("task2_stats.json not yet generated")
    with open(path) as f:
        stats = json.load(f)
    for key in ["arima_order", "lstm_window", "metrics", "best_model_by_rmse"]:
        assert key in stats, f"Missing key: {key}"


def test_task2_stats_metrics_have_both_models() -> None:
    """Metrics dict must contain entries for both ARIMA and LSTM."""
    path = DATA_DIR / "task2_stats.json"
    if not path.exists():
        pytest.skip("task2_stats.json not yet generated")
    with open(path) as f:
        stats = json.load(f)
    models = list(stats["metrics"].keys())
    assert any("ARIMA" in m for m in models), "ARIMA metrics missing"
    assert any("LSTM"  in m for m in models), "LSTM metrics missing"


# ── Figure tests ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("fname", [
    "t2_fig1_train_test_split.png",
    "t2_fig2_acf_pacf.png",
    "t2_fig3_arima_forecast.png",
    "t2_fig4_lstm_loss.png",
    "t2_fig5_lstm_forecast.png",
    "t2_fig6_model_comparison.png",
])
def test_task2_figures_exist(fname: str) -> None:
    """All Task 2 figures must be saved."""
    assert (IMG_DIR / fname).exists(), f"Missing figure: {fname}"
