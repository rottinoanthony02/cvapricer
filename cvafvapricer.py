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
    notional = st.number_input("Notional (USD)", 1e5, 1e11, 10_000_000.0, step=1e5, format="%.0f")
    T = st.number_input("Maturity (years)", 1.0, 50.0, 10.0, step=1.0)
    fixed_rate = st.number_input("Fixed rate K (decimal)", 0.0, 1.0, 0.03, step=0.0005, format="%.4f")
    st.caption("Annual cashflows for simplicity.")

    st.header("Market (flat short rate curve)")
    r0 = st.number_input("Initial rate r0", -1.0, 1.0, 0.03, step=0.0005, format="%.4f")
    sigma = st.number_input("Rate vol σ", 0.0001, 2.0, 0.01, step=0.0005, format="%.4f")

    st.header("Credit (CVA)")
    hazard = st.number_input("Hazard rate (annual, λ)", 0.0001, 1.0, 0.02, step=0.0005, format="%.4f")
    LGD = st.number_input("LGD", 0.0, 1.0, 0.6, step=0.05)

    st.header("Funding (FVA, non-collateralised)")
    s_f = st.number_input("Funding spread over OIS (decimal)", 0.0, 0.5, 0.01, step=0.0005, format="%.4f")
    st.caption("FVA = -∑ D(0,t) · ENE(t) · s_f · Δt, Δt=1y.")

    st.header("Simulation")
    n_paths = st.number_input("Monte Carlo paths", 1000, 500000, 50000, step=1000)
    seed = st.number_input("Random seed", 0, 10**9, 42)
    show_paths = st.slider("Show N rate paths on chart", 1, 200, 50)

    st.header("Stability options")
    clip_rates = st.checkbox("Clip simulated rates (avoid extremes)", value=True)
    r_min = st.number_input("Min rate clip", -1.0, 1.0, -0.05, step=0.005, format="%.3f", disabled=not clip_rates)
    r_max = st.number_input("Max rate clip", -1.0, 1.0, 0.20, step=0.005, format="%.3f", disabled=not clip_rates)

    st.header("Charts & Stats")
    show_epe_percentiles = st.checkbox("Show EPE percentiles (5/25/50/75/95)", value=True)
    horizons_str = st.text_input("Exposure histogram horizons (years, csv)", "1,5,10")
    bins = st.slider("Histogram bins", 10, 120, 40)

    compute_btn = st.button("Run")

st.markdown(
    "> **Pedagogical flat-curve model**: yearly steps, rates follow a simple Brownian walk, "
    "discount at r₀, constant hazard for CVA. FVA uses a constant funding spread on **negative exposure (ENE)**."
)

# ============ Core functions ============
def simulate_rates_flat(r0, sigma, T, n_paths, seed=42):
    """Annual steps (dt=1). r_t = r0 + sigma * sum N(0,1) * sqrt(dt)."""
    dt = 1.0
    n_steps = int(T / dt)
    rng = np.random.default_rng(int(seed))
    shocks = rng.standard_normal((n_paths, n_steps)) * np.sqrt(dt)
    rates = r0 + sigma * np.cumsum(shocks, axis=1)
    return rates  # (n_paths, n_steps) at t=1..n_steps

def swap_value_simple(notional, fixed_rate, rates, side="payer"):
    """
    Pathwise IRS value at each year t using flat-at-time-t short rate r_t:
      Float leg ≈ Σ_j DF(t,t+j) * r_t * N * dt
      Fixed leg  = Σ_j DF(t,t+j) * K   * N * dt
      DF(t,t+j)  = exp(-r_t * j), dt=1
    side="payer": value = float - fixed; "receiver": value = fixed - float.
    Overflow-safe via exponent clipping.
    """
    n_paths, n_steps = rates.shape
    V = np.zeros((n_paths, n_steps))
    for t in range(n_steps):
        remaining = n_steps - t
        if remaining <= 0:
            break
        j = np.arange(1, remaining + 1, dtype=float)

        # Prevent overflow/underflow in exp
        exp_mat = -np.outer(rates[:, t], j)        # shape (n_paths, remaining)
        np.clip(exp_mat, -700.0, 700.0, out=exp_mat)
        df = np.exp(exp_mat)

        float_leg = (df * rates[:, t][:, None]).sum(axis=1) * notional
        fixed_leg = (df * fixed_rate).sum(axis=1) * notional
        V[:, t] = (float_leg - fixed_leg) if side == "payer" else (fixed_leg - float_leg)
    return V

def exposures_from_values(V):
    """EPE from positive part; ENE from magnitude of negative part."""
    pos = np.maximum(V, 0.0)
    neg = np.maximum(-V, 0.0)   # magnitude
    EPE = pos.mean(axis=0)
    ENE = neg.mean(axis=0)
    return EPE, ENE, pos, neg

def cva_flat_curve(EPE, r0, hazard, LGD):
    """CVA = LGD * ∑ D(0,t) · EPE(t) · dPD(t) on yearly grid t=1..T."""
    n_steps = EPE.shape[0]
    times = np.arange(1, n_steps + 1, dtype=float)
    D = np.exp(-r0 * times)
    S = np.exp(-hazard * times)
    dPD = np.empty_like(times)
    dPD[0] = 1.0 - S[0]
    dPD[1:] = S[:-1] - S[1:]
    CVA = LGD * np.sum(D * EPE * dPD)
    return CVA, times, D, dPD

def fva_flat_curve(ENE, r0, s_f):
    """FVA (non-collateralised, constant spread): -∑ D(0,t) · ENE(t) · s_f · 1y."""
    n_steps = ENE.shape[0]
    times = np.arange(1, n_steps + 1, dtype=float)
    D = np.exp(-r0 * times)
    FVA = - np.sum(D * ENE * s_f * 1.0)
    return FVA

def epe_percentiles(paths_pos, q=(5,25,50,75,95)):
    times = np.arange(1, paths_pos.shape[1] + 1, dtype=float)
    qs = np.percentile(paths_pos, q, axis=0)
    data = {"t": times}
    for i, qi in enumerate(q):
        data[f"p{qi}"] = qs[i]
    return pd.DataFrame(data)

def parse_csv_floats(s):
    return [float(x.strip()) for x in s.split(",") if x.strip() != ""]

# ============ Run ============
if compute_btn:
    with st.spinner("Simulating and computing CVA & FVA..."):
        rates = simulate_rates_flat(r0, sigma, T, n_paths, seed=seed)
        if clip_rates:
            rates = np.clip(rates, r_min, r_max)

        side_key = "payer" if side.lower().startswith("payer") else "receiver"
        V = swap_value_simple(notional, fixed_rate, rates, side=side_key)
        EPE, ENE, paths_pos, paths_neg = exposures_from_values(V)

        CVA, times, D, dPD = cva_flat_curve(EPE, r0, hazard, LGD)
        FVA = fva_flat_curve(ENE, r0, s_f)
        XVA_total = CVA + FVA  # (no DVA/KVA here)

    # -------- Top results
    st.subheader("Results")
    m1, m2, m3 = st.columns(3)
    m1.metric("CVA (USD)", f"{CVA:,.2f}")
    m2.metric("FVA (USD)", f"{FVA:,.2f}")
    m3.metric("CVA + FVA (USD)", f"{XVA_total:,.2f}")
    st.caption(
        f"Side: **{side}**, Notional: **{notional:,.0f}**, K: **{fixed_rate:.4%}**, "
        f"T: **{T:.0f}y**, r0: **{r0:.2%}**, σ: **{sigma:.2%}**, λ: **{hazard:.2%}**, "
        f"LGD: **{LGD:.0%}**, funding spread s_f: **{s_f:.2%}**, clip={clip_rates}"
    )

    # -------- Charts
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("**Simulated rate paths (subset)**")
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
        st.markdown("**EPE & ENE**")
        df_e = pd.DataFrame({"t": times, "EPE": EPE, "ENE": ENE}).set_index("t")
        st.line_chart(df_e)
        st.caption("EPE(t) and ENE(t) in USD.")

    # -------- EPE percentiles (optional)
    if show_epe_percentiles:
        st.markdown("**EPE percentiles across paths**")
        epe_q_df = epe_percentiles(paths_pos, q=(5,25,50,75,95))
        st.line_chart(epe_q_df.set_index("t"))
        st.dataframe(epe_q_df, use_container_width=True)
        st.download_button(
            "⬇️ Download EPE percentiles (CSV)",
            epe_q_df.to_csv(index=False),
            file_name="epe_percentiles.csv"
        )

    # -------- Default & Discount charts
    c3, c4 = st.columns(2, gap="large")
    with c3:
        st.markdown("**Default increment dPD(t)**")
        df_dpd = pd.DataFrame({"t": times, "dPD": dPD})
        st.area_chart(df_dpd.set_index("t"))

    with c4:
        st.markdown("**Discount factor D(0,t)**")
        df_disc = pd.DataFrame({"t": times, "DF": D})
        st.line_chart(df_disc.set_index("t"))

    # -------- Histograms of exposure at selected horizons (EPE & ENE)
    st.subheader("Exposure histograms at selected horizons")
    try:
        horizons = parse_csv_floats(horizons_str)
    except Exception as e:
        horizons = []
        st.warning(f"Could not parse horizons: {e}")

    if horizons:
        horizons = sorted({int(h) for h in horizons if 1 <= int(h) <= int(T)})
        if not horizons:
            st.info("No valid horizons (must be integers between 1 and T).")
        else:
            cols = st.columns(min(3, len(horizons)))
            for idx, h in enumerate(horizons):
                expo_pos_h = paths_pos[:, h-1]  # positive exposure distribution
                expo_neg_h = paths_neg[:, h-1]  # negative exposure magnitude

                # Build histograms
                hist_pos, edges_pos = np.histogram(expo_pos_h, bins=bins)
                hist_neg, edges_neg = np.histogram(expo_neg_h, bins=bins)
                mid_pos = 0.5 * (edges_pos[1:] + edges_pos[:-1])
                mid_neg = 0.5 * (edges_neg[1:] + edges_neg[:-1])

                with cols[idx % len(cols)]:
                    st.markdown(f"**EPE distribution @ {h}y (positive exposure)**")
                    st.bar_chart(pd.DataFrame({"bucket": mid_pos, f"EPE@{h}y": hist_pos}).set_index("bucket"))
                    st.caption(
                        f"Mean={expo_pos_h.mean():,.0f}, p95={np.percentile(expo_pos_h,95):,.0f}"
                    )

                    st.markdown(f"**ENE distribution @ {h}y (negative exposure magnitude)**")
                    st.bar_chart(pd.DataFrame({"bucket": mid_neg, f"ENE@{h}y": hist_neg}).set_index("bucket"))
                    st.caption(
                        f"Mean={expo_neg_h.mean():,.0f}, p95={np.percentile(expo_neg_h,95):,.0f}"
                    )

    # -------- Tables / Downloads
    st.subheader("Data")
    out = pd.DataFrame({"t": times, "EPE": EPE, "ENE": ENE, "dPD": dPD, "DF": D})
    st.dataframe(out, use_container_width=True)
    st.download_button(
        "⬇️ Download EPE/ENE/dPD/DF (CSV)",
        out.to_csv(index=False),
        file_name="epe_ene_dpd_df.csv"
    )

    with st.expander("Notes / Model assumptions"):
        st.write(
            "- **CVA** uses constant hazard λ and LGD with yearly default increments.\n"
            "- **FVA (non-collateralised)** uses constant funding spread s_f on **negative exposure** (ENE):\n"
            "  FVA = -∑ D(0,t) · ENE(t) · s_f · Δt, Δt=1y. Sign convention: negative (cost).\n"
            "- Flat curve: discount at r₀; rates evolve as Brownian walk without mean reversion.\n"
            "- IRS value uses DF(t,t+j)=exp(-r_t·j) and annual cashflows; pedagogical proxy, not desk-grade."
        )
else:
    st.info("Adjust parameters in the sidebar and click **Run**.")
