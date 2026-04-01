"""
Implied Volatility Surface
==========================
Backs out implied volatility from market option prices using
Brent's root-finding method, then plots the volatility smile/surface.

Author: Armanbyek Soltanmurat
"""

import numpy as np
from scipy.optimize import brentq
from black_scholes import black_scholes


def implied_vol(market_price: float, S: float, K: float, T: float,
                r: float, option_type: str = "call",
                tol: float = 1e-6, max_iter: int = 500) -> float:
    """
    Compute implied volatility via Brent's method.

    Parameters
    ----------
    market_price : float       - Observed market option price
    S            : float       - Current stock price
    K            : float       - Strike price
    T            : float       - Time to expiry in years
    r            : float       - Risk-free rate
    option_type  : str         - "call" or "put"

    Returns
    -------
    float - Implied volatility (e.g. 0.20 for 20%)

    Raises
    ------
    ValueError if no IV solution found in [0.001, 10.0]
    """
    key = "call" if option_type == "call" else "put"

    def objective(sigma):
        return black_scholes(S, K, T, sigma, r)[key] - market_price

    # Intrinsic value bounds check
    intrinsic = max(S - K * np.exp(-r * T), 0) if key == "call" else max(K * np.exp(-r * T) - S, 0)
    if market_price < intrinsic - tol:
        raise ValueError(f"Market price ${market_price:.2f} is below intrinsic value ${intrinsic:.2f}.")

    try:
        iv = brentq(objective, 1e-4, 10.0, xtol=tol, maxiter=max_iter)
    except ValueError:
        raise ValueError(
            f"Could not find implied vol for K={K}, T={T}, price={market_price:.2f}. "
            "Check that the price is arbitrage-free."
        )
    return iv


def iv_smile(S: float, T: float, r: float, strikes: list[float],
             market_prices: list[float], option_type: str = "call") -> dict:
    """
    Compute the implied volatility smile across a range of strikes.

    Parameters
    ----------
    S             : float       - Current stock price
    T             : float       - Time to expiry in years
    r             : float       - Risk-free rate
    strikes       : list[float] - List of strike prices
    market_prices : list[float] - Corresponding market prices
    option_type   : str         - "call" or "put"

    Returns
    -------
    dict with keys: strikes, ivs, moneyness
    """
    ivs = []
    for K, price in zip(strikes, market_prices):
        try:
            ivs.append(implied_vol(price, S, K, T, r, option_type) * 100)
        except ValueError:
            ivs.append(np.nan)

    moneyness = [np.log(S / K) for K in strikes]   # log-moneyness
    return {"strikes": strikes, "ivs": ivs, "moneyness": moneyness, "T": T}


def synthetic_market_prices(S: float, strikes: list[float],
                              maturities: list[float], r: float,
                              base_vol: float = 0.20) -> dict:
    """
    Generate synthetic market prices with a realistic volatility skew.
    Useful for testing when real market data is unavailable.

    Skew model: sigma(K) = base_vol + skew * log(S/K) + smile * log(S/K)^2
    """
    skew_coeff  = 0.05
    smile_coeff = 0.02
    prices = {}
    for T in maturities:
        prices[T] = []
        for K in strikes:
            m = np.log(S / K)
            vol = max(0.01, base_vol + skew_coeff * m + smile_coeff * m**2)
            prices[T].append(black_scholes(S, K, T, vol, r)["call"])
    return prices


if __name__ == "__main__":
    S, r = 100, 0.05
    strikes    = [80, 85, 90, 95, 100, 105, 110, 115, 120]
    maturities = [0.25, 0.5, 1.0]

    print("=" * 55)
    print("  Implied Volatility Smile")
    print("=" * 55)

    for T in maturities:
        market_prices = synthetic_market_prices(S, strikes, [T], r)[T]
        smile = iv_smile(S, T, r, strikes, market_prices)
        print(f"\n  T = {T:.2f} years")
        print(f"  {'Strike':>8}  {'Market $':>10}  {'IV':>8}")
        for K, price, iv in zip(smile["strikes"], market_prices, smile["ivs"]):
            print(f"  {K:>8.0f}  ${price:>9.4f}  {iv:>7.2f}%")
    print("=" * 55)