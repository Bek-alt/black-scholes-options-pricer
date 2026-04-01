"""
Monte Carlo Option Pricing
==========================
Prices European options by simulating Geometric Brownian Motion paths
and compares results to the analytical Black-Scholes formula.

Author: Armanbyek Soltanmurat
"""

import numpy as np
from black_scholes import black_scholes


def simulate_gbm(S: float, T: float, sigma: float, r: float,
                 n_paths: int = 10_000, n_steps: int = 252,
                 seed: int | None = None) -> np.ndarray:
    """
    Simulate Geometric Brownian Motion paths.

    dS = S * (r*dt + sigma*sqrt(dt)*Z),  Z ~ N(0,1)

    Parameters
    ----------
    S        : float       - Initial stock price
    T        : float       - Time horizon in years
    sigma    : float       - Annualised volatility
    r        : float       - Risk-free rate
    n_paths  : int         - Number of Monte Carlo paths
    n_steps  : int         - Number of time steps per path
    seed     : int | None  - Random seed for reproducibility

    Returns
    -------
    np.ndarray of shape (n_paths,) with terminal stock prices S_T
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    # Antithetic variates for variance reduction
    Z = rng.standard_normal((n_paths // 2, n_steps))
    Z = np.vstack([Z, -Z])                          # shape: (n_paths, n_steps)
    log_returns = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    S_T = S * np.exp(log_returns.sum(axis=1))
    return S_T


def mc_price(S: float, K: float, T: float, sigma: float, r: float,
             n_paths: int = 10_000, n_steps: int = 252,
             seed: int | None = 42) -> dict:
    """
    Price a European call and put using Monte Carlo simulation.

    Returns
    -------
    dict with keys: mc_call, mc_put, bs_call, bs_put,
                    error_call, error_pct_call, std_call, n_paths
    """
    S_T = simulate_gbm(S, T, sigma, r, n_paths, n_steps, seed)
    discount = np.exp(-r * T)

    call_payoffs = np.maximum(S_T - K, 0)
    put_payoffs  = np.maximum(K - S_T, 0)

    mc_call = discount * call_payoffs.mean()
    mc_put  = discount * put_payoffs.mean()
    std_call = discount * call_payoffs.std() / np.sqrt(n_paths)   # std error

    bs = black_scholes(S, K, T, sigma, r)

    return {
        "mc_call":        mc_call,
        "mc_put":         mc_put,
        "bs_call":        bs["call"],
        "bs_put":         bs["put"],
        "error_call":     abs(mc_call - bs["call"]),
        "error_pct_call": abs(mc_call - bs["call"]) / bs["call"] * 100,
        "std_error_call": std_call,
        "n_paths":        n_paths,
    }


def convergence_analysis(S: float, K: float, T: float, sigma: float, r: float,
                          path_counts: list[int] | None = None) -> list[dict]:
    """
    Run MC pricing at increasing path counts to illustrate convergence.
    Useful for visualising the law of large numbers in a notebook.
    """
    if path_counts is None:
        path_counts = [100, 500, 1_000, 5_000, 10_000, 50_000, 100_000]

    results = []
    for n in path_counts:
        r_ = mc_price(S, K, T, sigma, r, n_paths=n, seed=None)
        results.append({"n_paths": n, "mc_call": r_["mc_call"],
                        "error": r_["error_call"]})
    return results


if __name__ == "__main__":
    params = dict(S=100, K=100, T=1.0, sigma=0.20, r=0.05)
    result = mc_price(**params, n_paths=10_000)

    print("=" * 45)
    print("  Monte Carlo vs Black-Scholes")
    print("=" * 45)
    print(f"  Paths simulated : {result['n_paths']:,}")
    print(f"  MC  call price  : ${result['mc_call']:.4f}")
    print(f"  BS  call price  : ${result['bs_call']:.4f}")
    print(f"  Absolute error  : ${result['error_call']:.4f}")
    print(f"  Relative error  : {result['error_pct_call']:.2f}%")
    print(f"  Std error (95%) : ±${1.96 * result['std_error_call']:.4f}")
    print("=" * 45)

    print("\nConvergence analysis:")
    print(f"  {'Paths':>8}  {'MC Call':>8}  {'Error':>8}")
    for row in convergence_analysis(**params):
        print(f"  {row['n_paths']:>8,}  ${row['mc_call']:>7.4f}  ${row['error']:>7.4f}")