"""
tests/test_task5.py
Unit tests for Task 5 – Backtesting outputs.

Verifies that the notebook outputs (JSON + images) exist, have the expected
structure, and contain plausible values. These tests run in CI after the
notebook has been executed and its outputs committed.
"""

import json
from pathlib import Path

import pytest

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE      = Path(__file__).parent.parent
DATA      = BASE / "data" / "processed"
IMAGES    = BASE / "notebooks" / "images"
JSON_PATH = DATA / "task5_stats.json"

PORTFOLIOS = ["Max Sharpe", "Min Volatility", "Benchmark 60/40"]
TICKERS    = ["TSLA", "BND", "SPY"]
METRICS    = ["Total Return", "CAGR", "Ann. Volatility", "Sharpe Ratio", "Max Drawdown"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_stats() -> dict:
    with open(JSON_PATH) as f:
        return json.load(f)


# ── Tests: JSON exists & top-level structure ──────────────────────────────────

class TestTask5StatsFile:

    def test_json_exists(self):
        assert JSON_PATH.exists(), f"Missing {JSON_PATH}"

    def test_top_level_keys(self):
        stats = load_stats()
        for key in ("backtest_period", "risk_free_rate", "portfolios"):
            assert key in stats, f"Missing top-level key: {key}"

    def test_backtest_period_keys(self):
        stats = load_stats()
        bp = stats["backtest_period"]
        assert "start" in bp and "end" in bp, "backtest_period missing start/end"

    def test_all_portfolios_present(self):
        stats = load_stats()
        for p in PORTFOLIOS:
            assert p in stats["portfolios"], f"Missing portfolio entry: {p}"


# ── Tests: Weights ────────────────────────────────────────────────────────────

class TestTask5Weights:

    @pytest.mark.parametrize("portfolio", PORTFOLIOS)
    def test_weights_sum_to_one(self, portfolio):
        stats = load_stats()
        w = stats["portfolios"][portfolio]["weights"]
        total = sum(w.values())
        assert abs(total - 1.0) < 1e-4, \
            f"{portfolio} weights sum to {total:.6f}, expected ~1.0"

    @pytest.mark.parametrize("portfolio", PORTFOLIOS)
    def test_weights_non_negative(self, portfolio):
        stats = load_stats()
        w = stats["portfolios"][portfolio]["weights"]
        for ticker, weight in w.items():
            assert weight >= -1e-6, \
                f"{portfolio} {ticker} weight is negative: {weight}"

    @pytest.mark.parametrize("portfolio", PORTFOLIOS)
    def test_all_tickers_in_weights(self, portfolio):
        stats = load_stats()
        w = stats["portfolios"][portfolio]["weights"]
        for t in TICKERS:
            assert t in w, f"{portfolio} missing weight for {t}"

    def test_benchmark_spy_weight(self):
        """Benchmark must be 60 % SPY."""
        stats = load_stats()
        w = stats["portfolios"]["Benchmark 60/40"]["weights"]
        assert abs(w["SPY"] - 0.60) < 1e-4, \
            f"Benchmark SPY weight expected 0.60, got {w['SPY']}"

    def test_benchmark_bnd_weight(self):
        """Benchmark must be 40 % BND."""
        stats = load_stats()
        w = stats["portfolios"]["Benchmark 60/40"]["weights"]
        assert abs(w["BND"] - 0.40) < 1e-4, \
            f"Benchmark BND weight expected 0.40, got {w['BND']}"


# ── Tests: Metrics structure & plausibility ───────────────────────────────────

class TestTask5Metrics:

    @pytest.mark.parametrize("portfolio", PORTFOLIOS)
    def test_all_metrics_present(self, portfolio):
        stats = load_stats()
        m = stats["portfolios"][portfolio]["metrics"]
        for key in METRICS:
            assert key in m, f"{portfolio} missing metric: {key}"

    @pytest.mark.parametrize("portfolio", PORTFOLIOS)
    def test_metrics_are_numbers(self, portfolio):
        stats = load_stats()
        m = stats["portfolios"][portfolio]["metrics"]
        for key, val in m.items():
            assert isinstance(val, (int, float)), \
                f"{portfolio}.{key} is not a number: {val}"

    @pytest.mark.parametrize("portfolio", PORTFOLIOS)
    def test_volatility_positive(self, portfolio):
        stats = load_stats()
        vol = stats["portfolios"][portfolio]["metrics"]["Ann. Volatility"]
        assert vol > 0, f"{portfolio} Ann. Volatility must be positive, got {vol}"

    @pytest.mark.parametrize("portfolio", PORTFOLIOS)
    def test_volatility_plausible(self, portfolio):
        """Annualised volatility should be between 0.5 % and 100 %."""
        stats = load_stats()
        vol = stats["portfolios"][portfolio]["metrics"]["Ann. Volatility"]
        assert 0.5 <= vol <= 100.0, \
            f"{portfolio} Ann. Volatility {vol:.2f}% is outside plausible range"

    @pytest.mark.parametrize("portfolio", PORTFOLIOS)
    def test_max_drawdown_non_positive(self, portfolio):
        """Max Drawdown is always <= 0 (expressed as a negative %)."""
        stats = load_stats()
        dd = stats["portfolios"][portfolio]["metrics"]["Max Drawdown"]
        assert dd <= 0.0, \
            f"{portfolio} Max Drawdown should be ≤ 0, got {dd}"

    def test_min_vol_lower_volatility_than_max_sharpe(self):
        """Min-Volatility portfolio must have lower annualised vol than Max-Sharpe."""
        stats = load_stats()
        vol_ms  = stats["portfolios"]["Max Sharpe"]["metrics"]["Ann. Volatility"]
        vol_mv  = stats["portfolios"]["Min Volatility"]["metrics"]["Ann. Volatility"]
        assert vol_mv <= vol_ms + 1e-4, \
            f"Min-Vol ({vol_mv:.2f}%) should be ≤ Max-Sharpe ({vol_ms:.2f}%)"


# ── Tests: Images ─────────────────────────────────────────────────────────────

class TestTask5Images:

    @pytest.mark.parametrize("fname", [
        "t5_fig1_cumulative_returns.png",
        "t5_fig2_drawdown.png",
        "t5_fig3_metrics_comparison.png",
    ])
    def test_image_exists(self, fname):
        path = IMAGES / fname
        assert path.exists(), f"Missing image: {path}"

    @pytest.mark.parametrize("fname", [
        "t5_fig1_cumulative_returns.png",
        "t5_fig2_drawdown.png",
        "t5_fig3_metrics_comparison.png",
    ])
    def test_image_nonzero_size(self, fname):
        path = IMAGES / fname
        if path.exists():
            assert path.stat().st_size > 5000, \
                f"Image too small (possibly empty): {path}"
