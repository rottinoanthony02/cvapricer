# cvapricer_multi_ccy_cva_cds.py
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="CVA – Multi-CCY with CDS Bootstrap (Flat Curve, Simple MC)", layout="wide")
st.title("💳 CVA – IRS (USD / EUR / GBP / CAD) – Flat curve MC + CDS hazard bootstrap")

# -----------------------------
# Currency presets (baseline params & fixed-leg conventions)
# -----------------------------
CCY_PRESETS = {
    "USD": {"symbol": "$",  "fixed_freq": 2, "fixed_dcc": "30/360 US",
            "r0": 0.030, "sigma": 0.010, "hazard": 0.020, "LGD": 0.60},
    "EUR": {"symbol": "€",  "fixed_freq": 1, "fixed_dcc": "30E/360",
            "r0": 0.020, "sigma": 0.010, "hazard": 0.020, "LGD": 0.60},
    "GBP": {"symbol": "£",  "fixed_freq": 2, "fixed_dcc": "ACT/365F",
            "r0": 0.035, "sigma": 0.012, "hazard": 0.020, "LGD": 0.60},
    "CAD": {"symbol": "C$", "fixed_freq": 2, "fixed_dcc": "ACT/365F",
            "r0": 0.030, "sigma": 0.011, "hazard": 0.020, "LGD": 0.60},
}

# -----------------------------
# Sidebar – instrument, market, credit, simulation, charts
# -----------------------------
with st.sidebar:
    st.header("Instrument")
    ccy = st.selectbox("Currency", list(CCY_PRESETS.keys()), index=0)
    p = CCY_PRESETS[ccy]
    sym = p["symbol"]

    side = st.selectbox("Swap side", ["Payer fixed", "Receiver fixed"], index=0)
    notional = st.number_input(f"Notional ({ccy})", 1e5, 1e12, 10_000_000.0, step=1e5, format="%.0f")
    maturity = st.number_input("Maturity (years)", 1.0, 50.0, 10.0, step=0.5)

    st.markdown("**Fixed-leg conventions**")
    fixed_freq = st.selectbox("Fixed payments per year", [1, 2, 4], index=[1,2,4].index(p["fixed_freq"]))
    fixed_dcc = st.selectbox("Fixed day count (label)", ["30/360 US", "30E/360", "ACT/365F"],
                             index=["30/360 US","30E/360","ACT/365F"].index(p["fixed_dcc"]))
    fixed_rate = st.number_input("Fixed rate K (decimal)", 0.0, 1.0, 0.03, step=0.0005, format="%.4f")
    st.caption("In this pedagogical grid, accrual τ ≈ 1/frequency (we don't compute exact day counts).")

    st.header("Market (flat short rate curve)")
    r0 = st.number_input("Initial rate r₀", -1.0, 1.0, float(p["r0"]), step=0.0005, format="%.4f")
    sigma = st.number_input("Rate vol σ", 0.0001, 2.0, float(p["sigma"]), step=0.0005, format="%.4f")

    st.header("Counterparty credit (CVA)")
    credit_mode = st.radio("Credit input", ["Constant hazard λ", "CDS curve bootstrap"], index=1)

    if credit_mode == "Constant hazard λ":
        hazard = st.number_input("Hazard rate λ (annual)", 0.0001, 1.0, float(p["hazard"]), step=0.0005, format="%.4f")
        LGD = st.number_input("LGD", 0.0, 1.0, float(p["LGD"]), step=0.05)
        recovery = 1.0 - LGD  # informational
    else:
        LGD = st.number_input("LGD", 0.0, 1.0, float(p["LGD"]), step=0.05)
        recovery = st.number_input("Recovery R (decimal)", 0.0, 1.0, 0.40, step=0.05, format="%.2f")
        st.caption("Enter CDS tenors and spreads (bps). Bootstrap uses quarterly premium payments.")
        cds_tenors_str = st.text_input("CDS tenors (years)", "1,3,5,7,10")
        cds_spreads_str = st.text_input("CDS spreads (bps)", "80,100,120,140,160")

    st.header("Simulation")
    n_paths = st.number_input("Monte Carlo paths", 1000, 500000, 50000, step=1000)
    seed = st.number_input("Random seed", 0, 10**9, 42)
    show_paths = st.slider("Show N rate paths", 1, 200, 50)
    clip_rates = st.checkbox("Clip simulated rates (avoid extremes)", value=True)
    r_min = st.number_input("Min rate clip", -1.0, 1.0, -0.05, step=0.005, format="%.3f", disabled=not clip_rates)
    r_max = st.number_input("Max rate clip", -1.0, 1.0, 0.20, step=0.005, format="%.3f", disabled=not clip_rates)

    st.header("Charts & Stats")
    show_epe_percentiles = st.checkbox("Show EPE percentiles (5/25/50/75/95)", value=True)

    compute_btn = st.button("Run CVA")

st.markdown(
    "> **Model**: flat curve at r₀; short rate is Brownian on the fixed-leg grid (Δt = 1/frequency). "
    "Exposure uses a simple IRS proxy with DF(t,t+mΔt)=exp(-r_t·m·Δt). Credit is either constant λ or "
    "bootstrapped piecewise hazard from CDS (quarterly payments, risky PV01 vs protection)."
)

# -----------------------------
# Utilities
# -----------------------------
def parse_csv_floats(s):
    return [float(x.strip()) for x in s.split(",") if x.strip() != ""]

def df0_flat(r0):
    return lambda t: float(np.exp(-r0 * float(t)))

# -----------------------------
# CDS Bootstrap (piecewise-constant hazard)
# -----------------------------
def cds_pay_grid(T, pay_freq=4):
    """Quarterly payment dates 0..T (inclusive)."""
    n = int(np.round(T * pay_freq))
    return np.linspace(0.0, T, n + 1)

def cds_leg_rpv01(times_pay, df0, S):
    """Risky PV01: sum tau * DF(0,t_i) * S(t_i)."""
    rpv01 = 0.0
    for i in range(1, len(times_pay)):
        tau = times_pay[i] - times_pay[i-1]
        t = times_pay[i]
        rpv01 += tau * df0(t) * S(t)
    return rpv01

def cds_leg_protection(times_pay, df0, S, R):
    """Protection leg: sum DF(0,t_i) * (S(t_{i-1}) - S(t_i)) * (1-R)."""
    prot = 0.0
    for i in range(1, len(times_pay)):
        t0, t1 = times_pay[i-1], times_pay[i]
        prot += df0(t1) * (S(t0) - S(t1)) * (1.0 - R)
    return prot

def bootstrap_piecewise_hazard(tenors, spreads_bps, df0, R=0.4, pay_freq=4,
                               max_iter=100, tol=1e-10):
    """
    Solve sequentially for λ_i on intervals [T_{i-1},T_i] s.t.
      spread_i * RPv01_i = Protection_i
    with S(t) piecewise-constant hazard (stepwise). Extrapolate flat hazard beyond last tenor.
    Returns hazards list (λ_i for each interval) and survival function S(t).
    """
    tenors = np.array(tenors, dtype=float)
    spreads = np.array(spreads_bps, dtype=float) * 1e-4  # decimal
    cut_times = np.concatenate([[0.0], tenors])
    hazards = []

    def S_with_hazards(t, hs):
        t = np.atleast_1d(t)
        Svals = np.ones_like(t, dtype=float)
        for j in range(1, len(cut_times)):
            lam = hs[j-1] if j-1 < len(hs) else (hs[-1] if len(hs) > 0 else 0.0)
            t0, t1 = cut_times[j-1], cut_times[j]
            dt = np.clip(np.minimum(t, t1) - t0, 0.0, None)
            Svals *= np.exp(-lam * dt)
        if len(hs) > 0:
            extra = np.clip(t - cut_times[-1], 0.0, None)
            Svals *= np.exp(-hs[-1] * extra)
        return Svals if Svals.shape != () else float(Svals)

    for i, T in enumerate(tenors):
        grid = cds_pay_grid(T, pay_freq=pay_freq)
        S_spread = spreads[i]

        def eq_for_lambda(lmb):
            hs_try = hazards + [lmb]
            S_try = lambda t: S_with_hazards(t, hs_try)
            return S_spread * cds_leg_rpv01(grid, df0, S_try) - cds_leg_protection(grid, df0, S_try, R)

        lo, hi = 1e-8, 5.0
        f_lo = eq_for_lambda(lo)
        mid = None
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            f_mid = eq_for_lambda(mid)
            if abs(f_mid) < tol:
                hazards.append(mid)
                break
            if np.sign(f_mid) == np.sign(f_lo):
                lo, f_lo = mid, f_mid
            else:
                hi = mid
        else:
            hazards.append(mid if mid is not None else lo)

    S_final = lambda t: S_with_hazards(t, hazards)
    return hazards, S_final

# -----------------------------
# Core CVA model (flat curve + simple IRS proxy)
# -----------------------------
def simulate_rates(r0, sigma, T, dt, n_paths, seed=42):
    """r_t = r0 + σ * ∑ N(0,1) * sqrt(dt) on grid Δt."""
    n_steps = int(np.ceil(T / dt))
    grid = np.linspace(dt, n_steps * dt, n_steps)   # t1..tN
    rng = np.random.default_rng(int(seed))
    shocks = rng.standard_normal((n_paths, n_steps)) * np.sqrt(dt)
    rates = r0 + sigma * np.cumsum(shocks, axis=1)
    return grid, rates  # times shape (n_steps,), rates (n_paths, n_steps)

def swap_value_paths(notional, K, rates, dt, side="payer"):
    """
    Pathwise IRS value V_k on grid {t_k}, using flat-at-time r_t per path:
      DF(t,t+m·dt) = exp(-r_t * m * dt)
      Float ≈ Σ_m DF * r_t * N * dt
      Fixed  = Σ_m DF * K   * N * dt
    side='payer' → value = float - fixed; 'receiver' → fixed - float.
    Overflow-safe via exponent clipping.
    """
    n_paths, n_steps = rates.shape
    V = np.zeros((n_paths, n_steps))
    for k in range(n_steps):
        remaining = n_steps - k
        if remaining <= 0:
            break
        m = np.arange(1, remaining + 1, dtype=float)
        # Prevent overflow/underflow in exp
        exp_mat = -np.outer(rates[:, k], m * dt)
        np.clip(exp_mat, -700.0, 700.0, out=exp_mat)
        df = np.exp(exp_mat)
        float_leg = (df * rates[:, k][:, None]).sum(axis=1) * notional * dt
        fixed_leg = (df * K).sum(axis=1) * notional * dt
        V[:, k] = (float_leg - fixed_leg) if side == "payer" else (fixed_leg - float_leg)
    return V

def exposures_from_values(V):
    """EPE from positive part only."""
    pos = np.maximum(V, 0.0)
    EPE = pos.mean(axis=0)
    return EPE, pos

def cva_from_EPE_constant_lambda(EPE, r0, hazard, grid, LGD):
    D = np.exp(-r0 * grid)
    S = np.exp(-hazard * grid)
    dPD = np.empty_like(grid)
    dPD[0] = 1.0 - S[0]
    dPD[1:] = S[:-1] - S[1:]
    CVA = LGD * float(np.sum(D * EPE * dPD))
    return D, S, dPD, CVA

def cva_from_EPE_bootstrapped(EPE, r0, grid, LGD, S_func):
    """
    Use bootstrapped survival S(t) to build discrete default increments on the exposure grid.
    """
    D = np.exp(-r0 * grid)
    S_vals = S_func(grid)
    dPD = np.empty_like(grid)
    dPD[0] = 1.0 - S_vals[0]
    dPD[1:] = S_vals[:-1] - S_vals[1:]
    CVA = LGD * float(np.sum(D * EPE * dPD))
    return D, S_vals, dPD, CVA

# -----------------------------
# Run
# -----------------------------
if compute_btn:
    # Time step aligned with fixed-leg frequency
    dt = 1.0 / float(fixed_freq)

    # Simulate rates
    grid, rates = simulate_rates(r0, sigma, maturity, dt, n_paths, seed=seed)
    if clip_rates:
        rates = np.clip(rates, r_min, r_max)

    side_key = "payer" if side.lower().startswith("payer") else "receiver"
    V = swap_value_paths(notional, fixed_rate, rates, dt, side=side_key)
    EPE, paths_pos = exposures_from_values(V)

    # Credit: constant λ or CDS bootstrap
    df0 = df0_flat(r0)

    hazards_list = None
    S = None
    if credit_mode == "Constant hazard λ":
        D, S_vals, dPD, CVA = cva_from_EPE_constant_lambda(EPE, r0, hazard, grid, LGD)
    else:
        # Parse CDS inputs
        try:
            cds_tenors = parse_csv_floats(cds_tenors_str)
            cds_spreads_bps = parse_csv_floats(cds_spreads_str)
        except Exception as e:
            st.error(f"Error parsing CDS inputs: {e}")
            st.stop()
        if len(cds_tenors) == 0 or len(cds_tenors) != len(cds_spreads_bps):
            st.error("CDS tenors/spreads must be non-empty and of the same length.")
            st.stop()

        hazards_list, S = bootstrap_piecewise_hazard(cds_tenors, cds_spreads_bps, df0, R=recovery, pay_freq=4)
        D, S_vals, dPD, CVA = cva_from_EPE_bootstrapped(EPE, r0, grid, LGD, S)

    # -------- Result
    st.subheader("Result")
    st.metric(f"CVA ({ccy})", f"{CVA:,.2f}")
    if credit_mode == "Constant hazard λ":
        cred_desc = f"λ: **{hazard:.2%}**, LGD: **{LGD:.0%}** (R≈{1-LGD:.0%})"
    else:
        cred_desc = f"CDS bootstrap, LGD: **{LGD:.0%}** (R={recovery:.0%})"
    st.caption(
        f"{ccy} {sym} | Side: **{side}** | Notional: **{notional:,.0f}** | K: **{fixed_rate:.4%}** | "
        f"T: **{maturity:.2f}y** | Freq: **{fixed_freq}x** (τ≈{dt:.3f}) [{fixed_dcc}] | "
        f"r₀: **{r0:.2%}**, σ: **{sigma:.2%}** | {cred_desc}"
    )

    # -------- Charts
    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown("**Simulated short-rate paths (subset)**")
        df_rates = pd.DataFrame({"t": grid})
        take = min(show_paths, rates.shape[0])
        for i in range(take):
            df_rates[f"path_{i+1}"] = rates[i]
        df_rates["mean_rate"] = rates.mean(axis=0)
        st.line_chart(df_rates.set_index("t"))
        st.caption(f"Δt = {dt:.3f} years (aligned to fixed-leg frequency).")

    with c2:
        st.markdown("**Expected Positive Exposure (EPE)**")
        st.line_chart(pd.DataFrame({"t": grid, "EPE": EPE}).set_index("t"))

    if show_epe_percentiles:
        st.markdown("**EPE percentiles across paths**")
        qs_df = pd.DataFrame({"t": grid})
        qs = np.percentile(paths_pos, [5,25,50,75,95], axis=0)
        for q, arr in zip([5,25,50,75,95], qs):
            qs_df[f"p{q}"] = arr
        st.line_chart(qs_df.set_index("t"))
        st.dataframe(qs_df, use_container_width=True)

    c3, c4 = st.columns(2, gap="large")
    with c3:
        st.markdown("**Default increment dPD(t)**")
        st.area_chart(pd.DataFrame({"t": grid, "dPD": dPD}).set_index("t"))
    with c4:
        st.markdown("**Discount factor D(0,t)**")
        st.line_chart(pd.DataFrame({"t": grid, "DF": np.exp(-r0 * grid)}).set_index("t"))

    # For CDS mode: show survival & hazards
    if credit_mode == "CDS curve bootstrap":
        st.subheader("Bootstrapped survival & hazards")
        surv_df = pd.DataFrame({"t": grid, "Survival S(t)": S_vals})
        st.line_chart(surv_df.set_index("t"))

        haz_df = pd.DataFrame({"TenorEnd(yrs)": cds_tenors, "Hazard λ_i": hazards_list})
        st.dataframe(haz_df, use_container_width=True)
        st.download_button("⬇️ Download hazards (CSV)", haz_df.to_csv(index=False), file_name=f"hazards_{ccy}.csv")

    # -------- Data / download
    st.subheader("CVA data")
    out = pd.DataFrame({"t": grid, "EPE": EPE, "dPD": dPD, "DF": np.exp(-r0 * grid)})
    st.dataframe(out, use_container_width=True)
    st.download_button("⬇️ Download EPE/dPD/DF (CSV)", out.to_csv(index=False), file_name=f"cva_data_{ccy}.csv")

    with st.expander("Notes / assumptions"):
        st.write(
            "- **CVA only**. Counterparty default via constant hazard λ or CDS-bootstrapped hazards; LGD supplied.\n"
            "- Flat discount at r₀; short-rate simulated as Brownian on the fixed-leg grid.\n"
            "- IRS value uses a simple proxy with DF(t,t+mΔt)=exp(-r_t·m·Δt) and accrual τ≈Δt.\n"
            "- **CDS bootstrap**: quarterly premium grid; solves S·RPV01 = protection per tenor (piecewise hazard). "
            "Beyond last tenor, hazard is kept flat (extrapolated)."
        )
else:
    st.info("Set parameters in the sidebar and click **Run CVA**.")
