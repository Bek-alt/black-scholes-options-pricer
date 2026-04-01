# Black-Scholes Options Pricer

A professional Python implementation of the **Black-Scholes option pricing model** with Greeks calculation, Monte Carlo simulation, and implied volatility extraction.

Built as a quantitative finance project demonstrating mathematical modelling, numerical methods, and software engineering best practices.

---

## Features

| Module | Description |
|---|---|
| `black_scholes.py` | Analytical pricing of European call & put options |
| `greeks.py` | Delta, Gamma, Vega, Theta, Rho (analytical) |
| `monte_carlo.py` | GBM simulation with antithetic variates |
| `implied_vol.py` | IV extraction via Brent's root-finding |
| `tests/` | 21 unit tests including finite-difference Greek validation |

---

## Installation
```bash
git clone https://github.com/Bek-alt/black-scholes-options-pricer
cd black-scholes-options-pricer
pip install -r requirements.txt
```

## Usage
```python
from src.black_scholes import black_scholes
from src.greeks import greeks

params = dict(S=100, K=100, T=1.0, sigma=0.20, r=0.05)
result = black_scholes(**params)
print(f"Call: ${result['call']:.4f}, Put: ${result['put']:.4f}")
```

## Running Tests
```bash
python -m pytest tests/ -v
```

## Key Results

| Metric | Value |
|---|---|
| ATM Call (S=K=100, T=1, σ=20%, r=5%) | **$10.4506** |
| Put-call parity error | **< 1e-10** |
| MC vs BS error (10k paths) | **< 0.5%** |
| IV round-trip error | **< 1e-5** |

---

## Author

Armanbyek Soltanmurat — MSc Mathematics student