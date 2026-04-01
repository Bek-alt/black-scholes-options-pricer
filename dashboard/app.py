"""
Black-Scholes Options Pricer — Streamlit Dashboard
====================================================
Interactive web app for pricing European options,
visualising Greeks, Monte Carlo simulation, and
the implied volatility surface.

Run with:  streamlit run dashboard/app.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from black_scholes import black_scholes
from greeks import greeks
from monte_carlo import mc_price, simulate_gbm
from implied_vol import implied_vol, synthetic_market_prices

# ─── Page config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Black-Scholes Pricer",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Black-Scholes Options Pricer")
st.caption("European option pricing · Greeks · Monte Carlo · Implied Volatility Surface")

# ─── Sidebar — Parameters ────────────────────────────────────────────────────

with st.sidebar:
    st.header("Parameters")
    S     = st.slider("Stock price (S)",       min_value=50.0,  max_value=200.0, value=100.0, step=1.0)
    K     = st.slider("Strike price (K)",      min_value=50.0,  max_value=200.0, value=100.0, step=1.0)
    T     = st.slider("Time to expiry (years)",min_value=0.1,   max_value=3.0,   value=1.0,   step=0.05)
    sigma = st.slider("Volatility (σ)",        min_value=0.01,  max_value=1.0,   value=0.20,  step=0.01, format="%.2f")
    r     = st.slider("Risk-free rate (r)",    min_value=0.0,   max_value=0.20,  value=0.05,  step=0.005, format="%.3f")
    st.divider()
    n_paths = st.select_slider("MC paths", options=[1_000, 5_000, 10_000, 50_000], value=10_000)

# ─── Compute ─────────────────────────────────────────────────────────────────

params = dict(S=S, K=K, T=T, sigma=sigma, r=r)
bs  = black_scholes(**params)
g   = greeks(**params)
mc  = mc_price(**params, n_paths=n_paths, seed=42)

# ─── Top metrics ─────────────────────────────────────────────────────────────

st.subheader("Option Prices")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Call price",  f"${bs['call']:.4f}")
c2.metric("Put price",   f"${bs['put']:.4f}")
c3.metric("d1",          f"{bs['d1']:.4f}")
c4.metric("d2",          f"{bs['d2']:.4f}")

st.divider()

# ─── Tabs ─────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs(["Greeks", "Price vs Stock", "Monte Carlo", "Implied Vol Surface"])

# ── Tab 1: Greeks ────────────────────────────────────────────────────────────

with tab1:
    st.subheader("Option Greeks")

    g1, g2, g3, g4, g5 = st.columns(5)
    g1.metric("Δ Delta (call)",  f"{g['delta_call']:.4f}")
    g2.metric("Γ Gamma",         f"{g['gamma']:.4f}")
    g3.metric("ν Vega",          f"{g['vega']:.4f}")
    g4.metric("Θ Theta (call)",  f"{g['theta_call']:.4f}")
    g5.metric("ρ Rho (call)",    f"{g['rho_call']:.4f}")

    st.markdown("#### Greeks vs Stock Price")

    stocks = np.linspace(50, 200, 200)
    delta_c = [greeks(s, K, T, sigma, r)["delta_call"] for s in stocks]
    delta_p = [greeks(s, K, T, sigma, r)["delta_put"]  for s in stocks]
    gamma_v = [greeks(s, K, T, sigma, r)["gamma"]      for s in stocks]
    vega_v  = [greeks(s, K, T, sigma, r)["vega"]       for s in stocks]
    theta_v = [greeks(s, K, T, sigma, r)["theta_call"] for s in stocks]

    greek_choice = st.selectbox("Select Greek to plot", ["Delta", "Gamma", "Vega", "Theta"])

    fig = go.Figure()
    if greek_choice == "Delta":
        fig.add_trace(go.Scatter(x=stocks, y=delta_c, name="Call Delta", line=dict(color="#1D9E75", width=2)))
        fig.add_trace(go.Scatter(x=stocks, y=delta_p, name="Put Delta",  line=dict(color="#D85A30", width=2)))
    elif greek_choice == "Gamma":
        fig.add_trace(go.Scatter(x=stocks, y=gamma_v, name="Gamma", line=dict(color="#534AB7", width=2)))
    elif greek_choice == "Vega":
        fig.add_trace(go.Scatter(x=stocks, y=vega_v,  name="Vega",  line=dict(color="#185FA5", width=2)))
    elif greek_choice == "Theta":
        fig.add_trace(go.Scatter(x=stocks, y=theta_v, name="Theta (call)", line=dict(color="#BA7517", width=2)))

    fig.add_vline(x=S, line_dash="dash", line_color="gray", annotation_text=f"S = {S}")
    fig.add_vline(x=K, line_dash="dot",  line_color="gray", annotation_text=f"K = {K}")
    fig.update_layout(xaxis_title="Stock price", yaxis_title=greek_choice,
                      height=400, template="plotly_white", legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)

# ── Tab 2: Price vs Stock ─────────────────────────────────────────────────────

with tab2:
    st.subheader("Option Price vs Stock Price")

    stocks = np.linspace(50, 200, 200)
    calls  = [black_scholes(s, K, T, sigma, r)["call"] for s in stocks]
    puts   = [black_scholes(s, K, T, sigma, r)["put"]  for s in stocks]
    intrinsic_call = [max(s - K, 0) for s in stocks]
    intrinsic_put  = [max(K - s, 0) for s in stocks]

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=stocks, y=calls, name="Call price",
                              line=dict(color="#1D9E75", width=2.5)))
    fig2.add_trace(go.Scatter(x=stocks, y=puts,  name="Put price",
                              line=dict(color="#D85A30", width=2.5)))
    fig2.add_trace(go.Scatter(x=stocks, y=intrinsic_call, name="Call intrinsic",
                              line=dict(color="#1D9E75", width=1, dash="dash")))
    fig2.add_trace(go.Scatter(x=stocks, y=intrinsic_put,  name="Put intrinsic",
                              line=dict(color="#D85A30", width=1, dash="dash")))
    fig2.add_vline(x=S, line_dash="dash", line_color="gray", annotation_text=f"Current S = {S}")
    fig2.add_vline(x=K, line_dash="dot",  line_color="gray", annotation_text=f"Strike K = {K}")
    fig2.update_layout(xaxis_title="Stock price", yaxis_title="Option price ($)",
                       height=450, template="plotly_white", legend=dict(orientation="h"))
    st.plotly_chart(fig2, use_container_width=True)

    st.info("Dashed lines show intrinsic value (payoff if exercised immediately). "
            "The gap between solid and dashed is the **time value** of the option.")

# ── Tab 3: Monte Carlo ────────────────────────────────────────────────────────

with tab3:
    st.subheader("Monte Carlo Simulation")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("MC Call price",   f"${mc['mc_call']:.4f}")
    m2.metric("BS Call price",   f"${mc['bs_call']:.4f}")
    m3.metric("Absolute error",  f"${mc['error_call']:.4f}")
    m4.metric("Relative error",  f"{mc['error_pct_call']:.3f}%")

    st.markdown("#### Simulated Stock Price Paths")
    n_display = st.slider("Paths to display", min_value=10, max_value=200, value=50)

    S_T = simulate_gbm(S, T, sigma, r, n_paths=n_display, n_steps=252, seed=0)
    steps = 252
    dt = T / steps
    rng = np.random.default_rng(0)
    Z = rng.standard_normal((n_display, steps))
    paths = np.zeros((n_display, steps + 1))
    paths[:, 0] = S
    for t in range(steps):
        paths[:, t+1] = paths[:, t] * np.exp(
            (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, t]
        )

    time_axis = np.linspace(0, T, steps + 1)
    fig3 = go.Figure()
    for i in range(n_display):
        color = "#1D9E75" if paths[i, -1] >= K else "#D85A30"
        fig3.add_trace(go.Scatter(x=time_axis, y=paths[i], mode="lines",
                                  line=dict(width=0.6, color=color),
                                  showlegend=False, opacity=0.5))
    fig3.add_hline(y=K, line_dash="dash", line_color="gray",
                   annotation_text=f"Strike K = {K}")
    fig3.update_layout(xaxis_title="Time (years)", yaxis_title="Stock price ($)",
                       height=450, template="plotly_white")
    st.plotly_chart(fig3, use_container_width=True)
    st.caption("Green paths = call finishes in-the-money. Red paths = call finishes out-of-the-money.")

    st.markdown("#### Terminal Price Distribution")
    S_T_full = simulate_gbm(S, T, sigma, r, n_paths=n_paths, seed=42)
    fig4 = px.histogram(S_T_full, nbins=80,
                        color_discrete_sequence=["#534AB7"],
                        labels={"value": "Terminal stock price"})
    fig4.add_vline(x=K, line_dash="dash", line_color="#D85A30",
                   annotation_text=f"Strike K = {K}")
    fig4.add_vline(x=float(np.mean(S_T_full)), line_dash="dot", line_color="#1D9E75",
                   annotation_text=f"Mean = ${np.mean(S_T_full):.1f}")
    fig4.update_layout(height=350, template="plotly_white", showlegend=False,
                       xaxis_title="Terminal stock price ($)", yaxis_title="Count")
    st.plotly_chart(fig4, use_container_width=True)

# ── Tab 4: Implied Vol Surface ────────────────────────────────────────────────

with tab4:
    st.subheader("Implied Volatility Surface")

    strikes    = np.linspace(70, 140, 15)
    maturities = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]

    market_prices = synthetic_market_prices(S, list(strikes), maturities, r, base_vol=sigma)

    iv_surface = []
    for T_ in maturities:
        row = []
        for i, strike in enumerate(strikes):
            try:
                iv = implied_vol(market_prices[T_][i], S, strike, T_, r) * 100
            except Exception:
                iv = np.nan
            row.append(iv)
        iv_surface.append(row)

    iv_surface = np.array(iv_surface)

    fig5 = go.Figure(data=[go.Surface(
        z=iv_surface,
        x=strikes,
        y=maturities,
        colorscale="Viridis",
        colorbar=dict(title="IV (%)")
    )])
    fig5.update_layout(
        scene=dict(
            xaxis_title="Strike (K)",
            yaxis_title="Maturity (T)",
            zaxis_title="Implied Vol (%)",
        ),
        height=550,
        template="plotly_white"
    )
    st.plotly_chart(fig5, use_container_width=True)

    st.markdown("#### Volatility Smile (fixed maturity)")
    chosen_T = st.select_slider("Select maturity", options=maturities, value=1.0)
    idx = maturities.index(chosen_T)
    smile_ivs = iv_surface[idx]

    fig6 = go.Figure()
    fig6.add_trace(go.Scatter(x=strikes, y=smile_ivs, mode="lines+markers",
                              line=dict(color="#534AB7", width=2.5),
                              marker=dict(size=6)))
    fig6.add_vline(x=S, line_dash="dash", line_color="gray",
                   annotation_text=f"ATM (S={S})")
    fig6.update_layout(xaxis_title="Strike (K)", yaxis_title="Implied volatility (%)",
                       height=350, template="plotly_white")
    st.plotly_chart(fig6, use_container_width=True)
    st.caption("The volatility smile shows that market-implied volatility varies by strike — "
               "a key deviation from the constant-vol assumption of Black-Scholes.")