"""
Black-Scholes Option Pricing Model
===================================
Prices European call and put options using the Black-Scholes formula.

Author: Armanbyek Soltanmurat 
"""

import numpy as np
from scipy.stats import norm


def black_scholes(S: float, K: float, T: float, sigma: float, r: float) -> dict:
    """
    Compute Black-Scholes price for European call and put options.

    Parameters
    ----------
    S     : float - Current stock price
    K     : float - Strike price
    T     : float - Time to expiry in years
    sigma : float - Annualised volatility (e.g. 0.20 for 20%)
    r     : float - Risk-free interest rate (e.g. 0.05 for 5%)

    Returns
    -------
    dict with keys: call, put, d1, d2
    """
    if T <= 0:
        raise ValueError("Time to expiry T must be positive.")
    if sigma <= 0:
        raise ValueError("Volatility sigma must be positive.")
    if S <= 0 or K <= 0:
        raise ValueError("Stock price S and strike K must be positive.")

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    call = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put  = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return {"call": call, "put": put, "d1": d1, "d2": d2}


def put_call_parity_check(call: float, put: float, S: float,
                           K: float, T: float, r: float,
                           tol: float = 1e-6) -> bool:
    """
    Verify put-call parity: C - P = S - K*e^(-rT)
    Returns True if satisfied within tolerance.
    """
    lhs = call - put
    rhs = S - K * np.exp(-r * T)
    return abs(lhs - rhs) < tol


if __name__ == "__main__":
    params = dict(S=100, K=100, T=1.0, sigma=0.20, r=0.05)
    result = black_scholes(**params)

    print("=" * 40)
    print("  Black-Scholes Option Pricer")
    print("=" * 40)
    for k, v in params.items():
        print(f"  {k:>6} = {v}")
    print("-" * 40)
    print(f"  Call price : ${result['call']:.4f}")
    print(f"  Put  price : ${result['put']:.4f}")
    print(f"  d1         : {result['d1']:.4f}")
    print(f"  d2         : {result['d2']:.4f}")
    print(f"  Put-call parity OK: {put_call_parity_check(result['call'], result['put'], **params)}")
    print("=" * 40)