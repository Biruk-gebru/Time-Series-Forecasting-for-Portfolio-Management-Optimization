"""
tests/test_task4.py
Unit tests for Task 4 – Portfolio Optimization outputs.

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
JSON_PATH = DATA / "task4_stats.json"

TICKERS = ["TSLA", "BND", "SPY"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_stats() -> dict:
    with open(JSON_PATH) as f:
        return json.load(f)


# ── Tests: JSON exists & top-level structure ──────────────────────────────────

class TestTask4StatsFile:

    def test_json_exists(self):
        assert JSON_PATH.exists(), f"Missing {JSON_PATH}"

    def test_top_level_keys(self):
        stats = load_stats()
        for key in ("risk_free_rate", "expected_returns",
                    "max_sharpe", "min_volatility", "covariance_matrix"):
            assert key in stats, f"Missing top-level key: {key}"

    def test_risk_free_rate_positive(self):
        stats = load_stats()
        assert 0 < stats["risk_free_rate"] < 0.15, \
            "risk_free_rate should be between 0 % and 15 %"


# ── Tests: Expected returns ───────────────────────────────────────────────────

class TestExpectedReturns:

    def test_all_tickers_present(self):
        stats = load_stats()
        for t in TICKERS:
            assert t in stats["expected_returns"], \
                f"Missing expected return for {t}"

    def test_returns_are_floats(self):
        stats = load_stats()
        for t, ret in stats["expected_returns"].items():
            assert isinstance(ret, (int, float)), \
                f"Expected return for {t} is not a number: {ret}"

    def test_bnd_spy_returns_within_plausible_range(self):
        """BND and SPY historical annualised returns should be between -20% and 50%."""
        stats = load_stats()
        for t in ("BND", "SPY"):
            ret = stats["expected_returns"][t]
            assert -0.20 <= ret <= 0.50, \
                f"{t} expected return {ret:.2%} is out of plausible range"


# ── Tests: Portfolio weights ──────────────────────────────────────────────────

class TestPortfolioWeights:

    @pytest.mark.parametrize("portfolio", ["max_sharpe", "min_volatility"])
    def test_weights_sum_to_one(self, portfolio):
        stats = load_stats()
        w = stats[portfolio]["weights"]
        total = sum(w.values())
        assert abs(total - 1.0) < 1e-4, \
            f"{portfolio} weights sum to {total:.6f}, expected ~1.0"

    @pytest.mark.parametrize("portfolio", ["max_sharpe", "min_volatility"])
    def test_weights_non_negative(self, portfolio):
        stats = load_stats()
        w = stats[portfolio]["weights"]
        for ticker, weight in w.items():
            assert weight >= -1e-6, \
                f"{portfolio} {ticker} weight is negative: {weight}"

    @pytest.mark.parametrize("portfolio", ["max_sharpe", "min_volatility"])
    def test_all_tickers_in_weights(self, portfolio):
        stats = load_stats()
        w = stats[portfolio]["weights"]
        for t in TICKERS:
            assert t in w, f"{portfolio} missing weight for {t}"


# ── Tests: Portfolio performance metrics ─────────────────────────────────────

class TestPortfolioPerformance:

    @pytest.mark.parametrize("portfolio", ["max_sharpe", "min_volatility"])
    def test_volatility_positive(self, portfolio):
        stats = load_stats()
        vol = stats[portfolio]["volatility"]
        assert vol > 0, f"{portfolio} volatility should be positive, got {vol}"

    @pytest.mark.parametrize("portfolio", ["max_sharpe", "min_volatility"])
    def test_volatility_plausible(self, portfolio):
        """Annual portfolio volatility should be between 0.5% and 100%."""
        stats = load_stats()
        vol = stats[portfolio]["volatility"]
        assert 0.005 < vol < 1.0, \
            f"{portfolio} volatility {vol:.2%} is out of plausible range"

    def test_min_vol_lower_than_max_sharpe(self):
        """Min-volatility portfolio must have lower or equal volatility."""
        stats = load_stats()
        assert stats["min_volatility"]["volatility"] <= \
               stats["max_sharpe"]["volatility"] + 1e-6, \
            "Min-volatility portfolio has higher volatility than Max-Sharpe"

    def test_max_sharpe_has_keys(self):
        stats = load_stats()
        for key in ("weights", "expected_return", "volatility", "sharpe_ratio"):
            assert key in stats["max_sharpe"], \
                f"max_sharpe missing key: {key}"

    def test_min_vol_has_keys(self):
        stats = load_stats()
        for key in ("weights", "expected_return", "volatility", "sharpe_ratio"):
            assert key in stats["min_volatility"], \
                f"min_volatility missing key: {key}"


# ── Tests: Covariance matrix ──────────────────────────────────────────────────

class TestCovarianceMatrix:

    def test_all_tickers_present(self):
        stats = load_stats()
        cov = stats["covariance_matrix"]
        for t in TICKERS:
            assert t in cov, f"Covariance matrix missing row for {t}"
            for t2 in TICKERS:
                assert t2 in cov[t], f"Covariance matrix missing ({t}, {t2})"

    def test_diagonal_positive(self):
        """Variances (diagonal) must be positive."""
        stats = load_stats()
        cov = stats["covariance_matrix"]
        for t in TICKERS:
            assert cov[t][t] > 0, f"Variance for {t} is not positive"

    def test_symmetric(self):
        """Covariance matrix must be symmetric: cov[i][j] == cov[j][i]."""
        stats = load_stats()
        cov = stats["covariance_matrix"]
        for t in TICKERS:
            for t2 in TICKERS:
                assert abs(cov[t][t2] - cov[t2][t]) < 1e-8, \
                    f"Covariance matrix not symmetric at ({t}, {t2})"

    def test_tsla_higher_variance_than_bnd(self):
        """TSLA is much more volatile than BND."""
        stats = load_stats()
        cov = stats["covariance_matrix"]
        assert cov["TSLA"]["TSLA"] > cov["BND"]["BND"], \
            "Expected TSLA variance > BND variance"


# ── Tests: Images ─────────────────────────────────────────────────────────────

class TestTask4Images:

    @pytest.mark.parametrize("fname", [
        "t4_fig1_efficient_frontier.png",
        "t4_fig2_portfolio_weights.png",
    ])
    def test_image_exists(self, fname):
        path = IMAGES / fname
        assert path.exists(), f"Missing image: {path}"

    @pytest.mark.parametrize("fname", [
        "t4_fig1_efficient_frontier.png",
        "t4_fig2_portfolio_weights.png",
    ])
    def test_image_nonzero_size(self, fname):
        path = IMAGES / fname
        if path.exists():
            assert path.stat().st_size > 5000, \
                f"Image too small (possibly empty): {path}"
