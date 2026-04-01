"""
Unit Tests — Black-Scholes Options Pricer
==========================================
Tests cover:
  - Black-Scholes formula correctness
  - Put-call parity
  - Greeks analytical vs numerical (finite difference)
  - Monte Carlo convergence
  - Implied volatility round-trip

Run with:  python -m pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest
from black_scholes import black_scholes, put_call_parity_check
from greeks import greeks
from monte_carlo import mc_price
from implied_vol import implied_vol


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def atm():
    """At-the-money parameters."""
    return dict(S=100, K=100, T=1.0, sigma=0.20, r=0.05)

@pytest.fixture
def itm():
    """In-the-money call."""
    return dict(S=110, K=100, T=1.0, sigma=0.20, r=0.05)

@pytest.fixture
def otm():
    """Out-of-the-money call."""
    return dict(S=90, K=100, T=1.0, sigma=0.20, r=0.05)


# ─── Black-Scholes Tests ─────────────────────────────────────────────────────

class TestBlackScholes:
    def test_known_atm_call(self, atm):
        """ATM call ~10.45 (textbook value)."""
        result = black_scholes(**atm)
        assert abs(result["call"] - 10.4506) < 0.01

    def test_known_atm_put(self, atm):
        """ATM put ~5.57 (textbook value)."""
        result = black_scholes(**atm)
        assert abs(result["put"] - 5.5735) < 0.01

    def test_put_call_parity(self, atm):
        res = black_scholes(**atm)
        assert put_call_parity_check(res["call"], res["put"],
                                    S=atm["S"], K=atm["K"],
                                    T=atm["T"], r=atm["r"])
    def test_put_call_parity_itm(self, itm):
        res = black_scholes(**itm)
        assert put_call_parity_check(res["call"], res["put"],
                                    S=itm["S"], K=itm["K"],
                                    T=itm["T"], r=itm["r"])
    def test_call_increases_with_stock_price(self):
        """Call price is monotonically increasing in S."""
        calls = [black_scholes(S, 100, 1.0, 0.20, 0.05)["call"]
                 for S in range(80, 130, 5)]
        assert all(calls[i] < calls[i+1] for i in range(len(calls)-1))

    def test_call_bounded_below_by_zero(self, otm):
        result = black_scholes(**otm)
        assert result["call"] >= 0

    def test_invalid_T_raises(self, atm):
        atm["T"] = -1
        with pytest.raises(ValueError):
            black_scholes(**atm)

    def test_invalid_sigma_raises(self, atm):
        atm["sigma"] = 0
        with pytest.raises(ValueError):
            black_scholes(**atm)


# ─── Greeks Tests ────────────────────────────────────────────────────────────

class TestGreeks:
    EPS = 0.001   # finite difference step

    def fd_delta_call(self, params):
        up   = black_scholes(**{**params, "S": params["S"] + self.EPS})["call"]
        down = black_scholes(**{**params, "S": params["S"] - self.EPS})["call"]
        return (up - down) / (2 * self.EPS)

    def fd_gamma(self, params):
        up   = black_scholes(**{**params, "S": params["S"] + self.EPS})["call"]
        base = black_scholes(**params)["call"]
        down = black_scholes(**{**params, "S": params["S"] - self.EPS})["call"]
        return (up - 2*base + down) / self.EPS**2

    def test_delta_call_bounded(self, atm):
        g = greeks(**atm)
        assert 0 < g["delta_call"] < 1

    def test_delta_put_bounded(self, atm):
        g = greeks(**atm)
        assert -1 < g["delta_put"] < 0

    def test_delta_equals_finite_difference(self, atm):
        g = greeks(**atm)
        fd = self.fd_delta_call(atm)
        assert abs(g["delta_call"] - fd) < 1e-4

    def test_gamma_equals_finite_difference(self, atm):
        g = greeks(**atm)
        fd = self.fd_gamma(atm)
        assert abs(g["gamma"] - fd) < 1e-4

    def test_gamma_positive(self, atm):
        g = greeks(**atm)
        assert g["gamma"] > 0

    def test_vega_positive(self, atm):
        g = greeks(**atm)
        assert g["vega"] > 0

    def test_theta_negative_for_call(self, atm):
        """Time decay should reduce call value."""
        g = greeks(**atm)
        assert g["theta_call"] < 0


# ─── Monte Carlo Tests ───────────────────────────────────────────────────────

class TestMonteCarlo:
    def test_mc_close_to_bs(self, atm):
        """With 50k paths, MC should be within 2% of BS."""
        result = mc_price(**atm, n_paths=50_000, seed=42)
        assert result["error_pct_call"] < 2.0

    def test_mc_call_non_negative(self, atm):
        result = mc_price(**atm, n_paths=1_000, seed=0)
        assert result["mc_call"] >= 0

    def test_mc_put_call_parity(self, atm):
        """MC call and put should satisfy put-call parity approximately."""
        res = mc_price(**atm, n_paths=50_000, seed=42)
        lhs = res["mc_call"] - res["mc_put"]
        rhs = atm["S"] - atm["K"] * np.exp(-atm["r"] * atm["T"])
        assert abs(lhs - rhs) < 0.5   # loose tolerance for MC


# ─── Implied Volatility Tests ─────────────────────────────────────────────────

class TestImpliedVol:
    def test_iv_round_trip(self, atm):
        """Back out IV from a BS price — should recover original sigma."""
        price = black_scholes(**atm)["call"]
        iv = implied_vol(price, atm["S"], atm["K"], atm["T"], atm["r"])
        assert abs(iv - atm["sigma"]) < 1e-5

    def test_iv_round_trip_put(self, atm):
        price = black_scholes(**atm)["put"]
        iv = implied_vol(price, atm["S"], atm["K"], atm["T"], atm["r"], "put")
        assert abs(iv - atm["sigma"]) < 1e-5

    def test_iv_increases_with_vol(self):
        """Higher input vol → higher option price → higher recovered IV."""
        vols = [0.10, 0.20, 0.30, 0.40]
        prices = [black_scholes(100, 100, 1.0, v, 0.05)["call"] for v in vols]
        ivs = [implied_vol(p, 100, 100, 1.0, 0.05) for p in prices]
        assert all(ivs[i] < ivs[i+1] for i in range(len(ivs)-1))