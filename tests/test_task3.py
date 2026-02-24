"""
tests/test_task3.py
Unit tests for Task 3 – Future Forecast outputs.

Verifies that the notebook outputs (CSV + JSON) exist, have the expected
structure, and contain plausible values.  These tests run in CI after the
notebook has been executed and its outputs committed.
"""

import json
from pathlib import Path
import pandas as pd
import pytest

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE      = Path(__file__).parent.parent
DATA      = BASE / "data" / "processed"
IMAGES    = BASE / "notebooks" / "images"
CSV_PATH  = DATA / "task3_future_forecast.csv"
JSON_PATH = DATA / "task3_stats.json"

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_forecast_csv() -> pd.DataFrame:
    return pd.read_csv(CSV_PATH, index_col="Date", parse_dates=True)


def load_stats() -> dict:
    with open(JSON_PATH) as f:
        return json.load(f)


# ── Tests: CSV ────────────────────────────────────────────────────────────────

class TestForecastCSV:

    def test_csv_exists(self):
        assert CSV_PATH.exists(), f"Missing {CSV_PATH}"

    def test_csv_has_expected_columns(self):
        df = load_forecast_csv()
        expected = {"median_forecast", "mean_forecast", "p5", "p25", "p75", "p95"}
        assert expected.issubset(set(df.columns)), \
            f"Missing columns: {expected - set(df.columns)}"

    def test_csv_row_count(self):
        df = load_forecast_csv()
        # 12 months ~ 252 business days; allow a small margin
        assert 240 <= len(df) <= 265, \
            f"Expected ~252 rows, got {len(df)}"

    def test_no_nan_values(self):
        df = load_forecast_csv()
        assert not df.isnull().any().any(), "CSV contains NaN values"

    def test_ci_ordering(self):
        """p5 <= p25 <= median <= p75 <= p95 for all rows."""
        df = load_forecast_csv()
        assert (df["p5"] <= df["p25"]).all(), "p5 > p25 detected"
        assert (df["p25"] <= df["median_forecast"]).all(), "p25 > median detected"
        assert (df["median_forecast"] <= df["p75"]).all(), "median > p75 detected"
        assert (df["p75"] <= df["p95"]).all(), "p75 > p95 detected"

    def test_forecast_prices_positive(self):
        df = load_forecast_csv()
        assert (df["median_forecast"] > 0).all(), "Negative forecast prices found"

    def test_index_is_business_days(self):
        df = load_forecast_csv()
        # Day-of-week: 0=Mon … 4=Fri
        weekdays = df.index.dayofweek
        assert (weekdays < 5).all(), "Non-business day found in forecast index"


# ── Tests: JSON stats ─────────────────────────────────────────────────────────

class TestTask3Stats:

    def test_json_exists(self):
        assert JSON_PATH.exists(), f"Missing {JSON_PATH}"

    def test_json_top_level_keys(self):
        stats = load_stats()
        for key in ("model", "training_data", "forecast_params",
                    "last_known_price", "6m_forecast", "12m_forecast"):
            assert key in stats, f"Missing key: {key}"

    def test_model_is_lstm(self):
        stats = load_stats()
        assert stats["model"] == "LSTM"

    def test_last_known_price_positive(self):
        stats = load_stats()
        assert stats["last_known_price"] > 0

    def test_6m_forecast_keys(self):
        stats = load_stats()
        fm = stats["6m_forecast"]
        for key in ("date", "median_price", "p5_price", "p95_price", "expected_return_pct"):
            assert key in fm, f"Missing 6m_forecast key: {key}"

    def test_12m_forecast_keys(self):
        stats = load_stats()
        fm = stats["12m_forecast"]
        for key in ("date", "median_price", "p5_price", "p95_price", "expected_return_pct"):
            assert key in fm, f"Missing 12m_forecast key: {key}"

    def test_12m_ci_widens_vs_6m(self):
        """CI spread at 12m should be at least as wide as at 6m."""
        stats = load_stats()
        spread_6m  = stats["6m_forecast"]["p95_price"]  - stats["6m_forecast"]["p5_price"]
        spread_12m = stats["12m_forecast"]["p95_price"] - stats["12m_forecast"]["p5_price"]
        assert spread_12m >= spread_6m * 0.9, \
            "12m CI is unexpectedly narrower than 6m CI"


# ── Tests: Images ─────────────────────────────────────────────────────────────

class TestTask3Images:

    @pytest.mark.parametrize("fname", [
        "t3_fig1_6m_forecast.png",
        "t3_fig2_12m_forecast.png",
        "t3_fig3_trend_analysis.png",
    ])
    def test_image_exists(self, fname):
        path = IMAGES / fname
        assert path.exists(), f"Missing image: {path}"

    @pytest.mark.parametrize("fname", [
        "t3_fig1_6m_forecast.png",
        "t3_fig2_12m_forecast.png",
        "t3_fig3_trend_analysis.png",
    ])
    def test_image_nonzero_size(self, fname):
        path = IMAGES / fname
        if path.exists():
            assert path.stat().st_size > 5000, \
                f"Image too small (possibly empty): {path}"
