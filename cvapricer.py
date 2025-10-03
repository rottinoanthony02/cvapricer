# cvapricer_flat.py
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="CVA Monte Carlo – Flat Curve (Simple)", layout="wide")
st.title("💳 CVA Monte Carlo – IRS payer fixed (Flat curve, simple MC)")

# ============ Sidebar inputs ============
with st.sidebar:
    st.header("Swap")
    notional = st.number_input("Notional (USD)", 1e5, 1e10, 10_000_000.0, step=1e5, format="%.0f")
    T = st.number_input("Maturity (years)", 1.0, 50.0, 10.0, step=1.0)
    fixed_rate = st.number_input("Fixed rate K (decimal)", 0.0, 1.0, 0.03, step=0.0005, format="%.4f")
    st.caption("Payments assumed annual for simplicity.")

    st.header("Market (flat short rate curve)")
    r0 = st.number_input("Initial rate r0", 0.0, 1.0, 0.03, step=0.0005, format="%.4f")
    sigma = st.number_input("Rate vol σ", 0.0001, 1.0, 0.01, step=0.0005, format="%.4f")

    st.header("Credit / CVA")
    hazard = st.number_input("Hazard rate (annual, λ)", 0.0001, 1.0, 0.02, step=0.0005, format="%.4f")
    LGD = st.number_input("LGD", 0.0, 1.0, 0.6, step=0.05)

    st.header("Simulation")
    n_paths = st.number_input("Monte Carlo paths", 1000, 300000, 50000, step=1000)
    seed = st.number_input("Random seed", 0, 10**9, 42)
    show_paths = st.slider("Show N rate paths on chart", 1, 200, 50)
    compute_btn = st.button("Run CVA")

st.markdown(
    "> This is the **simple pedagogical version**: flat curve, yearly steps, "
    "rates follow a basic Brownian walk, exposure uses a simple IRS approximation."
)

# ============ Core functions ============
def simulate_rates_flat(r0, sigma, T, n_paths, seed=42):
    dt = 1.0  # annual
    n_steps = int(T / dt)
    rng = np.random.default_rng(int(seed))
    # r_t = r0 + sigma * sqrt(dt) * cumulative sum of N(0,1)
    shocks = rng.standard_normal((n_paths, n_steps)) * np.sqrt(dt)
    rates = r0 + sigma * np.cumsum(shocks, axis=1)
    return rates  # shape: (n_paths, n_steps), times 1..n_steps (years)

def swap_exposure_simple(notional, fixed_rate, rates):
    """
    At time step t (1..n_steps), value IRS payer-fixed ~ max(float - fixed, 0)
    Float leg ≈ sum_{j=1..remaining} DF(t, t+j) * r_t * notional * dt
    Fixed leg = sum_{j=1..remaining} DF(t, t+j) * K * notional * dt
    DF uses exp(-r_t * j) with flat-at-time-t rate.
    """
    n_paths, n_steps = rates.shape
    exposures = np.zeros((n_paths, n_steps))
    dt = 1.0
    for t in range(n_steps):
        remaining = n_steps - t
        if remaining <= 0:
            break
        # DF(t, t+j) with flat r_t on each path
        j = np.arange(1, remaining + 1)
        df = np.exp(-np.outer(rates[:, t], j))  # (n_paths, remaining)
        float_leg = (df * rates[:, t][:, None]).sum(axis=1) * notional * dt
        fixed_leg = (df * fixed_rate).sum(axis=1) * notional * dt
        exposures[:, t] = np.maximum(float_leg - fixed_leg, 0.0)
    return exposures  # (n_paths, n_steps)

def cva_flat_curve(exposures, r0, hazard, LGD):
    """
    EE(t) = mean positive exposure at year t.
    Discount D(0,t) = exp(-r0 * t). (flat)
    dPD(t-1,t) = S(t-1)-S(t) with S(t) = exp(-hazard*t).
    """
    n_steps = exposures.shape[1]
    times = np.arange(1, n_steps + 1, dtype=float)  # 1..n_steps (years)
    EE = exposures.mean(axis=0)

    D = np.exp(-r0 * times)
    S = np.exp(-hazard * times)
    dPD = np.empty_like(times)
    dPD[0] = 1.0 - S[0]
    dPD[1:] = S[:-1] - S[1:]

    CVA = LGD * np.sum(D * EE * dPD)
    return CVA, times, EE, D, dPD

# ============ Run ============
if compute_btn:
    with st.spinner("Simulating and pricing..."):
        rates = simulate_rates_flat(r0, sigma, T, n_paths, seed=seed)
        exposures = swap_exposure_simple(notional, fixed_rate, rates)
        CVA, times, EE, D, dPD = cva_flat_curve(exposures, r0, hazard, LGD)

    # -------- Top result
    st.subheader("Result")
    st.metric("CVA (USD)", f"{CVA:,.2f}")

    # -------- Charts
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("**Simulated rate paths (subset)**")
        # Build a small dataframe of first `show_paths` paths (plus mean)
        n_steps = rates.shape[1]
        t_grid = np.arange(1, n_steps + 1, dtype=float)
        df_rates = pd.DataFrame({"t": t_grid})
        take = min(show_paths, rates.shape[0])
        for i in range(take):
            df_rates[f"path_{i+1}"] = rates[i]
        df_rates["mean_rate"] = rates.mean(axis=0)
        st.line_chart(df_rates.set_index("t"))
        st.caption("Annual steps (t = 1..T).")

    with c2:
        st.markdown("**Expected Positive Exposure (EPE)**")
        df_epe = pd.DataFrame({"t": times, "EPE": EE})
        st.line_chart(df_epe.set_index("t"))
        st.caption("EPE(t) in USD")

    # -------- Secondary charts
    c3, c4 = st.columns(2, gap="large")
    with c3:
        st.markdown("**Default increment dPD(t)**")
        df_dpd = pd.DataFrame({"t": times, "dPD": dPD})
        st.area_chart(df_dpd.set_index("t"))

    with c4:
        st.markdown("**Discount factor D(0,t)**")
        df_disc = pd.DataFrame({"t": times, "DF": D})
        st.line_chart(df_disc.set_index("t"))

    # -------- Tables / Downloads
    st.subheader("Data")
    out = pd.DataFrame({"t": times, "EPE": EE, "dPD": dPD, "DF": D})
    st.dataframe(out, use_container_width=True)
    st.download_button("⬇️ Download EPE/dPD/DF (CSV)", out.to_csv(index=False), file_name="epe_dpd_df.csv")

    with st.expander("Notes / Model assumptions"):
        st.write(
            "- Flat curve approximation: discount uses constant r0; survival uses constant hazard λ.\n"
            "- Rates follow a simple Brownian walk with σ; no mean reversion.\n"
            "- Exposure of payer-fixed IRS uses pathwise **DF = exp(-r_t * j)** and a simple "
            "float ≈ sum DF·r_t and fixed ≈ sum DF·K; annual cashflows.\n"
            "- For stability, increase paths; CVA converges roughly as 1/√N."
        )
else:
    st.info("Adjust parameters in the sidebar and click **Run CVA**.")
