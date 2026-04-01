"""
Option Greeks
=============
Calculates Delta, Gamma, Vega, Theta, and Rho for European options.

All Greeks use the analytical Black-Scholes formulas.

Author: Armanbyek Soltanmurat
"""

import numpy as np
from scipy.stats import norm
from black_scholes import black_scholes


def greeks(S: float, K: float, T: float, sigma: float, r: float) -> dict:
    """
    Compute all first-order Greeks for European call and put options.

    Parameters
    ----------
    S     : float - Current stock price
    K     : float - Strike price
    T     : float - Time to expiry in years
    sigma : float - Annualised volatility
    r     : float - Risk-free rate

    Returns
    -------
    dict with keys: delta_call, delta_put, gamma, vega, theta_call,
                    theta_put, rho_call, rho_put
    """
    res = black_scholes(S, K, T, sigma, r)
    d1, d2 = res["d1"], res["d2"]

    # Delta — sensitivity of option price to stock price
    delta_call =  norm.cdf(d1)
    delta_put  =  norm.cdf(d1) - 1            # equivalent to -N(-d1)

    # Gamma — rate of change of delta (same for call and put)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))

    # Vega — sensitivity to volatility (per 1% move, divide by 100)
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100

    # Theta — time decay (per calendar day, divide by 365)
    theta_call = (
        -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
        - r * K * np.exp(-r * T) * norm.cdf(d2)
    ) / 365

    theta_put = (
        -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
        + r * K * np.exp(-r * T) * norm.cdf(-d2)
    ) / 365

    # Rho — sensitivity to interest rate (per 1% move, divide by 100)
    rho_call =  K * T * np.exp(-r * T) * norm.cdf(d2)  / 100
    rho_put  = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

    return {
        "delta_call": delta_call,
        "delta_put":  delta_put,
        "gamma":      gamma,
        "vega":       vega,
        "theta_call": theta_call,
        "theta_put":  theta_put,
        "rho_call":   rho_call,
        "rho_put":    rho_put,
    }


if __name__ == "__main__":
    params = dict(S=100, K=100, T=1.0, sigma=0.20, r=0.05)
    g = greeks(**params)

    print("=" * 40)
    print("  Option Greeks")
    print("=" * 40)
    print(f"  Delta (call) : {g['delta_call']:.4f}")
    print(f"  Delta (put)  : {g['delta_put']:.4f}")
    print(f"  Gamma        : {g['gamma']:.4f}")
    print(f"  Vega         : {g['vega']:.4f}  (per 1% vol move)")
    print(f"  Theta (call) : {g['theta_call']:.4f}  (per calendar day)")
    print(f"  Theta (put)  : {g['theta_put']:.4f}  (per calendar day)")
    print(f"  Rho   (call) : {g['rho_call']:.4f}  (per 1% rate move)")
    print(f"  Rho   (put)  : {g['rho_put']:.4f}  (per 1% rate move)")
    print("=" * 40)