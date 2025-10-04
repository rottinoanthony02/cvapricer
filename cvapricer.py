# cvapricer_multi_ccy_cva_cds_hw.py
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="CVA – Multi-CCY with CDS Bootstrap (Hull–White 1F)", layout="wide")
st.title("💳 CVA – IRS (USD / EUR / GBP / CAD) – Hull–White 1F rates + CDS hazard bootstrap")

# -----------------------------
# Currency presets (baseline params & fixed-leg conventions)
# -----------------------------
CCY_PRESETS = {
    "USD": {"symbol": "$",  "fixed_freq": 2, "fixed_dcc": "30/360 US",
            "r0": 0.030, "a": 0.05, "sigma": 0.010, "hazard": 0.020, "LGD": 0.60},
    "EUR": {"symbol": "€",  "fixed_freq": 1, "fixed_dcc": "30E/360",
            "r0": 0.020, "a": 0.05, "sigma": 0.010, "hazard": 0.020, "LGD": 0.60},
    "GBP": {"symbol": "£",  "fixed_freq": 2, "fixed_dcc": "ACT/365F",
            "r0": 0.035, "a": 0.05, "sigma": 0.012, "hazard": 0.020, "LGD": 0.60},
    "CAD": {"symbol": "C$", "fixed_freq": 2, "fixed_dcc": "ACT/365F",
            "r0": 0.030, "a": 0.05, "sigma": 0.011, "hazard": 0.020, "LGD": 0.60},
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
    st.caption("Accrual τ ≈ 1/frequency (simple grid).")

    st.header("Initial discount curve (for HW fit)")
    curve_mode = st.radio("Curve input", ["Flat r₀", "Custom zero curve"], index=0)
    if curve_mode == "Flat r₀":
        r0 = st.number_input("Flat short rate r₀", -1.0, 1.0, float(p["r0"]), step=0.0005, format="%.4f")
        zero_times = [0.0, maturity]
        zero_rates = [r0, r0]
    else:
        st.caption("Enter comma-separated year tenors and matching zero rates (decimals).")
        zero_times_str = st.text_input("Zero tenors (years)", "0,1,2,5,10,30")
        zero_rates_str = st.text_input("Zero rates (decimals)", "0.02,0.022,0.023,0.024,0.025,0.026")
        def parse_csv_floats(s): return [float(x.strip()) for x in s.split(",") if x.strip()]
        try:
            zero_times = parse_csv_floats(zero_times_str)
            zero_rates = parse_csv_floats(zero_rates_str)
        except Exception as e:
            st.error(f"Parse error for zero curve: {e}")
            st.stop()
        if len(zero_times) != len(zero_rates) or len(zero_times) < 2:
            st.error("Zero curve lists must have same length (≥2).")
            st.stop()
        r0 = float(zero_rates[0])

    st.header("Hull–White (1F)")
    a = st.number_input("Mean reversion a", 0.0001, 2.0, float(p["a"]), step=0.005, format="%.4f")
    sigma = st.number_input("Volatility σ", 0.0001, 2.0, float(p["sigma"]), step=0.0005, format="%.4f")

    st.header("Counterparty credit (CVA)")
    credit_mode = st.radio("Credit input", ["Constant hazard λ", "CDS curve bootstrap"], index=1)
    if credit_mode == "Constant hazard λ":
        hazard = st.number_input("Hazard rate λ (annual)", 0.0001, 1.0, float(p["hazard"]), step=0.0005, format="%.4f")
        LGD = st.number_input("LGD", 0.0, 1.0, float(p["LGD"]), step=0.05)
        recovery = 1.0 - LGD
    else:
        LGD = st.number_input("LGD", 0.0, 1.0, float(p["LGD"]), step=0.05)
        recovery = st.number_input("Recovery R (decimal)", 0.0, 1.0, 0.40, step=0.05, format="%.2f")
        st.caption("Enter CDS tenors and spreads (bps). Quarterly premium payments.")
        cds_tenors_str = st.text_input("CDS tenors (years)", "1,3,5,7,10")
        cds_spreads_str = st.text_input("CDS spreads (bps)", "80,100,120,140,160")

    st.header("Simulation")
    n_paths = st.number_input("Monte Carlo paths", 1000, 300000, 50000, step=1000)
    seed = st.number_input("Random seed", 0, 10**9, 123)
    show_paths = st.slider("Show N rate paths", 1, 200, 50)

    st.header("Charts & Stats")
    show_epe_percentiles = st.checkbox("Show EPE percentiles (5/25/50/75/95)", value=True)

    compute_btn = st.button("Run CVA")

st.markdown(
    "> **Model**: Hull–White 1F short rate fitted to your initial curve; pathwise DF from the rate integral. "
    "Credit is either constant λ or CDS-bootstrapped hazards (quarterly premiums). IRS exposure uses DF from HW paths."
)

# -----------------------------
# Curve helpers
# -----------------------------
def interp_zero(times, zeros):
    times = np.array(times, float); zeros = np.array(zeros, float)
    def z(t):
        t = float(t)
        if t <= times[0]: return float(zeros[0])
        if t >= times[-1]: return float(zeros[-1])
        i = np.searchsorted(times, t) - 1
        t0,t1 = times[i], times[i+1]; z0,z1 = zeros[i], zeros[i+1]
        w = (t - t0)/(t1 - t0)
        return float(z0*(1-w) + z1*w)
    return z

def forward_from_zero(times, zeros):
    z = interp_zero(times, zeros)
    def f0(t, h=1e-5):
        # instantaneous forward f(0,t) ≈ z(t) + t z'(t)
        zt = z(t); zp = (z(t+h)-z(t-h))/(2*h)
        return float(zt + t*zp)
    return f0

def df0_from_zero(times, zeros):
    z = interp_zero(times, zeros)
    return lambda t: float(np.exp(-z(t)*t))

# -----------------------------
# Hull–White 1F
# -----------------------------
def theta_from_f0_grid(grid, a, sigma, f0):
    h = 1e-4
    dfdt = (np.array([f0(t+h) for t in grid]) - np.array([f0(t-h) for t in grid]))/(2*h)
    return dfdt + a*np.array([f0(t) for t in grid]) + (sigma**2/(2*a))*(1.0 - np.exp(-2*a*grid))

def simulate_hw_paths(a, sigma, f0, T, dt, n_paths, seed=123):
    n_steps = int(np.ceil(T/dt))
    grid = np.linspace(0.0, n_steps*dt, n_steps+1)  # include t=0
    rng = np.random.default_rng(int(seed))
    r = np.zeros((n_paths, n_steps+1))
    r[:,0] = f0(1e-6)
    A = np.zeros_like(r)  # integral A_t

    theta_vals = theta_from_f0_grid(grid, a, sigma, f0)
    for k in range(n_steps):
        dW = rng.standard_normal(n_paths)*np.sqrt(dt)
        drift = theta_vals[k] - a*r[:,k]
        r[:,k+1] = r[:,k] + drift*dt + sigma*dW
        A[:,k+1] = A[:,k] + 0.5*(r[:,k] + r[:,k+1])*dt  # trapezoid

    return grid, r, A

def df_path(A, idx_from, idx_to):
    return np.exp(-(A[:, idx_to] - A[:, idx_from]))

# -----------------------------
# CDS bootstrap (piecewise-constant hazards)
# -----------------------------
def parse_csv_floats(s): return [float(x.strip()) for x in s.split(",") if x.strip()]

def cds_pay_grid(T, pay_freq=4):
    n = int(np.round(T * pay_freq))
    return np.linspace(0.0, T, n+1)

def cds_leg_rpv01(times_pay, df0, S):
    rpv01 = 0.0
    for i in range(1, len(times_pay)):
        tau = times_pay[i] - times_pay[i-1]
        t = times_pay[i]
        rpv01 += tau * df0(t) * S(t)
    return rpv01

def cds_leg_protection(times_pay, df0, S, R):
    prot = 0.0
    for i in range(1, len(times_pay)):
        t0, t1 = times_pay[i-1], times_pay[i]
        prot += df0(t1) * (S(t0) - S(t1)) * (1.0 - R)
    return prot

def bootstrap_piecewise_hazard(tenors, spreads_bps, df0, R=0.4, pay_freq=4, max_iter=100, tol=1e-10):
    tenors = np.array(tenors, float)
    spreads = np.array(spreads_bps, float) * 1e-4
    cut_times = np.concatenate([[0.0], tenors])
    hazards = []

    def S_with_hs(t, hs):
        t = np.atleast_1d(t)
        Svals = np.ones_like(t, dtype=float)
        for j in range(1, len(cut_times)):
            lam = hs[j-1] if j-1 < len(hs) else (hs[-1] if len(hs)>0 else 0.0)
            t0,t1 = cut_times[j-1], cut_times[j]
            dt = np.clip(np.minimum(t, t1) - t0, 0, None)
            Svals *= np.exp(-lam * dt)
        if len(hs)>0:
            extra = np.clip(t - cut_times[-1], 0, None)
            Svals *= np.exp(-hs[-1] * extra)
        return Svals if Svals.shape != () else float(Svals)

    for i, T in enumerate(tenors):
        grid = cds_pay_grid(T, 4)
        s = spreads[i]

        def f_of_lambda(lmb):
            hs_try = hazards + [lmb]
            S_try = lambda t: S_with_hs(t, hs_try)
            return s * cds_leg_rpv01(grid, df0, S_try) - cds_leg_protection(grid, df0, S_try, R)

        lo, hi = 1e-8, 5.0
        f_lo = f_of_lambda(lo)
        mid = None
        for _ in range(max_iter):
            mid = 0.5*(lo+hi)
            f_mid = f_of_lambda(mid)
            if abs(f_mid) < tol:
                hazards.append(mid)
                break
            if np.sign(f_mid) == np.sign(f_lo):
                lo, f_lo = mid, f_mid
            else:
                hi = mid
        else:
            hazards.append(mid if mid is not None else lo)

    S_final = lambda t: S_with_hs(t, hazards)
    return hazards, S_final

# -----------------------------
# Exposure & CVA on HW paths
# -----------------------------
def grid_indices_for_payments(grid, pay_times):
    # nearest indices on HW grid for each payment time
    idx = np.searchsorted(grid, pay_times, side="left")
    idx = np.clip(idx, 1, len(grid)-1)  # avoid t=0
    return idx.tolist()

def swap_exposure_hw(notional, K, grid, A_paths, pay_times, side="payer"):
    """
    Exposure at each HW time k using DF from A_paths:
      PV_float(t_k) ≈ N * (1 - P(t_k, T_last))
      PV_fixed(t_k) = N * Σ τ_i * K * P(t_k, t_i) over remaining fixed dates
    """
    n_paths, n_steps_plus = A_paths.shape
    exposures = np.zeros((n_paths, n_steps_plus))

    pay_idx = grid_indices_for_payments(grid, pay_times)
    last_idx = pay_idx[-1]

    for k in range(n_steps_plus):
        future_idx = [j for j in pay_idx if j > k]
        if not future_idx:
            continue
        # Float leg ≈ N*(1 - P(t_k, T_last))
        P_t_T = df_path(A_paths, k, last_idx)
        pv_float = notional * (1.0 - P_t_T)

        # Fixed leg
        pv_fixed = np.zeros(n_paths)
        prev = k
        for j in future_idx:
            tau = grid[j] - grid[prev]
            df = df_path(A_paths, k, j)
            pv_fixed += notional * K * tau * df
            prev = j

        value = (pv_float - pv_fixed) if side=="payer" else (pv_fixed - pv_float)
        exposures[:, k] = np.maximum(value, 0.0)

    return exposures  # includes t=0 column (k=0)

def cva_from_epe_grid(EPE, df0, S, grid, LGD):
    # grid includes t=0; drop k=0 for default increment calc
    t = grid[1:]
    E = EPE[1:]
    D = np.array([df0(tt) for tt in t])
    S_vals = S(t)
    dPD = np.empty_like(t)
    dPD[0] = 1.0 - S_vals[0]
    dPD[1:] = S_vals[:-1] - S_vals[1:]
    CVA = LGD * float(np.sum(D * E * dPD))
    return CVA, t, E, D, dPD, S_vals

# -----------------------------
# Run
# -----------------------------
if compute_btn:
    # Build curve & forwards for HW fit
    f0 = forward_from_zero(zero_times, zero_rates)
    df0 = df0_from_zero(zero_times, zero_rates)

    # HW time step aligned with fixed leg
    dt = 1.0 / float(fixed_freq)

    # Simulate HW short-rate & integral
    grid, r_paths, A_paths = simulate_hw_paths(a, sigma, f0, maturity, dt, n_paths, seed=seed)

    # Build fixed payment schedule on (0,T] with frequency
    n_pay = int(np.round(maturity * fixed_freq))
    pay_times = np.linspace(dt, n_pay*dt, n_pay)

    # Payer/Receiver
    side_key = "payer" if side.lower().startswith("payer") else "receiver"

    # Pathwise exposures using HW discounting
    exposures = swap_exposure_hw(notional, fixed_rate, grid, A_paths, pay_times, side=side_key)
    EPE = exposures.mean(axis=0)

    # Credit: constant λ or CDS bootstrap
    if credit_mode == "Constant hazard λ":
        S = lambda t: np.exp(-hazard * np.atleast_1d(t))
    else:
        # Parse CDS inputs and bootstrap hazards
        try:
            cds_tenors = parse_csv_floats(cds_tenors_str)
            cds_spreads_bps = parse_csv_floats(cds_spreads_str)
        except Exception as e:
            st.error(f"Error parsing CDS inputs: {e}")
            st.stop()
        if len(cds_tenors)==0 or len(cds_tenors)!=len(cds_spreads_bps):
            st.error("CDS tenors/spreads must be non-empty and of equal length.")
            st.stop()
        hazards_list, S = bootstrap_piecewise_hazard(cds_tenors, cds_spreads_bps, df0, R=recovery, pay_freq=4)

    # CVA on the HW grid (skip t=0)
    CVA, t_epe, E_epe, D0t, dPD, S_vals = cva_from_epe_grid(EPE, df0, S, grid, LGD)

    # -------- Result
    st.subheader("Result")
    st.metric(f"CVA ({ccy})", f"{CVA:,.2f}")
    cred_desc = (f"λ: **{hazard:.2%}**, LGD: **{LGD:.0%}** (R≈{1-LGD:.0%})"
                 if credit_mode=="Constant hazard λ"
                 else f"CDS bootstrap, LGD: **{LGD:.0%}** (R={recovery:.0%})")
    st.caption(
        f"{ccy} {sym} | Side: **{side}** | Notional: **{notional:,.0f}** | K: **{fixed_rate:.4%}** | "
        f"T: **{maturity:.2f}y** | Fixed freq: **{fixed_freq}x** (τ≈{dt:.3f}) [{fixed_dcc}] | "
        f"HW: a={a:.3f}, σ={sigma:.3%} | Curve: {'flat' if curve_mode=='Flat r₀' else 'custom zeros'} | {cred_desc}"
    )

    # -------- Charts
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("**Hull–White short-rate paths (subset)**")
        # show k>=1 (exclude t=0)
        t_plot = grid
        df_rates = pd.DataFrame({"t": t_plot})
        take = min(show_paths, r_paths.shape[0])
        for i in range(take):
            df_rates[f"path_{i+1}"] = r_paths[i]
        df_rates["mean_rate"] = r_paths.mean(axis=0)
        st.line_chart(df_rates.set_index("t"))
        st.caption(f"Δt = {dt:.3f} years.")

    with c2:
        st.markdown("**EPE on HW grid**")
        st.line_chart(pd.DataFrame({"t": grid, "EPE": EPE}).set_index("t"))

    if show_epe_percentiles:
        st.markdown("**EPE percentiles across paths**")
        pos_paths = np.maximum(exposures, 0.0)
        qs = np.percentile(pos_paths, [5,25,50,75,95], axis=0)
        qs_df = pd.DataFrame({"t": grid, "p5": qs[0], "p25": qs[1], "p50": qs[2], "p75": qs[3], "p95": qs[4]})
        st.line_chart(qs_df.set_index("t"))
        st.dataframe(qs_df.iloc[1:], use_container_width=True)  # skip t=0 in table

    c3, c4 = st.columns(2, gap="large")
    with c3:
        st.markdown("**Default increment dPD(t_k)**")
        st.area_chart(pd.DataFrame({"t": t_epe, "dPD": dPD}).set_index("t"))
    with c4:
        st.markdown("**Discount factor D(0,t)** (from initial curve)")
        st.line_chart(pd.DataFrame({"t": t_epe, "DF": D0t}).set_index("t"))

    if credit_mode == "CDS curve bootstrap":
        st.subheader("Bootstrapped survival & hazards")
        st.line_chart(pd.DataFrame({"t": t_epe, "Survival S(t)": S_vals}).set_index("t"))
        haz_df = pd.DataFrame({"TenorEnd(yrs)": cds_tenors, "Hazard λ_i": hazards_list})
        st.dataframe(haz_df, use_container_width=True)
        st.download_button("⬇️ Download hazards (CSV)", haz_df.to_csv(index=False), file_name=f"hazards_{ccy}.csv")

    # -------- Data / download
    st.subheader("CVA data")
    out = pd.DataFrame({"t": t_epe, "EPE": E_epe, "dPD": dPD, "DF": D0t})
    st.dataframe(out, use_container_width=True)
    st.download_button("⬇️ Download EPE/dPD/DF (CSV)", out.to_csv(index=False), file_name=f"cva_data_{ccy}.csv")

    with st.expander("Notes / assumptions"):
        st.write(
            "- **Rates**: Hull–White 1F with θ(t) fitted to the initial forward curve; pathwise DF from the short-rate integral.\n"
            "- **Exposure**: float leg ≈ N·(1−P(t,T_last)); fixed leg via pathwise discounting to remaining fixed dates.\n"
            "- **Credit**: constant hazard λ or CDS-bootstrapped piecewise hazards (quarterly premiums).\n"
            "- **Curve**: choose flat r₀ or custom zero curve; HW fit uses f(0,t) constructed from the zeros."
        )
else:
    st.info("Set parameters in the sidebar and click **Run CVA**.")
