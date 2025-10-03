# cvapricer_flat.py
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="CVA & FVA – Flat Curve (Simple)", layout="wide")
st.title("💳 CVA & FVA – IRS (Flat curve, simple MC)")

# ============ Sidebar inputs ============
with st.sidebar:
    st.header("Swap")
    side = st.selectbox("Swap side", ["Payer fixed", "Receiver fixed"], index=0)
    notional = st.number_input("Notional (USD)", 1e5, 1e10, 10_000_000.0, step=1e5, format="%.0f")
    T = st.number_input("Maturity (years)", 1.0, 50.0, 10.0, step=1.0)
    fixed_rate = st.number_input("Fixed rate K (decimal)", 0.0, 1.0, 0.03, step=0.0005, format="%.4f")
    st.caption("Payments assumed annual for simplicity.")

    st.header("Market (flat short rate curve)")
    r0 = st.number_input("Initial rate r0", 0.0, 1.0, 0.03, step=0.0005, format="%.4f")
    sigma = st.number_input("Rate vol σ", 0.0001, 1.0, 0.01, step=0.0005, format="%.4f")

    st.header("Credit (CVA)")
    hazard = st.number_input("Hazard rate (annual, λ)", 0.0001, 1.0, 0.02, step=0.0005, format="%.4f")
    LGD = st.number_input("LGD", 0.0, 1.0, 0.6, step=0.05)

    st.header("Funding (FVA, non-collateralised)")
    s_f = st.number_input("Funding spread over OIS (decimal)", 0.0, 0.2, 0.01, step=0.0005, format="%.4f")
    st.caption("Simple model: FVA = -∑ D(0,t) · ENE(t) · s_f · Δt, with Δt=1y.")

    st.header("Simulation")
    n_paths = st.number_input("Monte Carlo paths", 1000, 300000, 50000, step=1000)
    seed = st.number_input("Random seed", 0, 10**9, 42)
    show_paths = st.slider("Show N rate paths on chart", 1, 200, 50)

    st.header("Charts & Stats")
    show_epe_percentiles = st.checkbox("Show EPE percentiles (5/25/50/75/95)", value=True)
    horizons_str = st.text_input("Exposure histogram horizons (years, csv)", "1,5,10")
    bins = st.slider("Histogram bins", 10, 100, 40)

    compute_btn = st.button("Run")

st.markdown(
    "> **Pedagogical flat-curve model**: yearly steps, rates follow a simple Brownian walk, "
    "discount at r₀, constant hazard for CVA. FVA uses a constant funding spread on **negative exposure (ENE)**."
)

# ============ Core functions ============
def simulate_rates_flat(r0, sigma, T, n_paths, seed=42):
    dt = 1.0  # annual
    n_steps = int(T / dt)
    rng = np.random.default_rng(int(seed))
    shocks = rng.standard_normal((n_paths, n_steps)) * np.sqrt(dt)
    rates = r0 + sigma * np.cumsum(shocks, axis=1)
    return rates  # (n_paths, n_steps), times 1..n_steps

def swap_value_simple(notional, fixed_rate, rates, side="payer"):
    """
    Pathwise IRS value at each year t using flat-at-time-t short rate r_t:
      Float leg ≈ sum_j DF(t,t+j) * r_t * N * dt
      Fixed leg  = sum_j DF(t,t+j) * K   * N * dt
      DF(t,t+j)  = exp(-r_t * j), dt=1
    side="payer": value = float - fixed
    side="receiver": value = fixed - float
    Returns V (n_paths, n_steps).
    """
    n_paths, n_steps = rates.shape
    V = np.zeros((n_paths, n_steps))
    for t in range(n_steps):
        remaining = n_steps - t
        if remaining <= 0:
            break
        j = np.arange(1, remaining + 1)
        df = np.exp(-np.outer(rates[:, t], j))  # (n_paths, remaining)
        float_leg = (df * rates[:, t][:, None]).sum(axis=1) * notional
        fixed_leg = (df * fixed_rate).sum(axis=1) * notional
        if side == "payer":
            V[:, t] = float_leg - fixed_leg
        else:
            V[:, t] = fixed_leg - float_leg
    return V

def exposures_from_values(V):
    """
    EPE uses positive part; ENE uses negative part (as positive magnitude).
    Returns (EPE, ENE, EE_pos_paths, EE_neg_paths) where EE_neg_paths are magnitudes.
    """
    pos = np.maximum(V, 0.0)
    neg = np.maximum(-V, 0.0)   # magnitude of negative value
    EPE = pos.mean(axis=0)
    ENE = neg.mean(axis=0)
    return EPE, ENE, pos, neg

def cva_flat_curve(EPE, r0, hazard, LGD):
    """
    CVA = LGD * ∑ D(0,t) · EPE(t) · dPD(t*
