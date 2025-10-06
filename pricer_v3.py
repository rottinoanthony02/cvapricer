# pricer_v4.py
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Swap Par • CVA • Heston Barrier", layout="wide")
st.title("Swap Par • CVA (CSA) • Heston Barrier — by A. Rottino")

# -----------------------------
# Currency presets
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
# Helpers: curve, HW, CVA, CSA, Heston, Barrier
# -----------------------------
def interp_zero(times, zeros):
    times = np.asarray(times, float)
    zeros = np.asarray(zeros, float)
    def z(t):
        t_arr = np.asarray(t, float)
        return np.interp(t_arr, times, zeros, left=zeros[0], right=zeros[-1])
    return z

def forward_from_zero(times, zeros):
    z = interp_zero(times, zeros)
    def f0(t):
        t_arr = np.asarray(t, float)
        h = 1e-5
        zt = z(t_arr)
        zp = (z(t_arr + h) - z(t_arr - h)) / (2 * h)
        return zt + t_arr * zp
    return f0

def df0_from_zero(times, zeros):
    z = interp_zero(times, zeros)
    def df(t):
        t_arr = np.asarray(t, float)
        return np.exp(-z(t_arr) * t_arr)
    return df

def simulate_hw_paths(a, sigma, f0, T, dt, n_paths, seed=123):
    n_steps = int(np.ceil(T / dt))
    grid = np.linspace(0, n_steps * dt, n_steps + 1)
    rng = np.random.default_rng(int(seed))
    r = np.zeros((n_paths, n_steps + 1))
    A = np.zeros_like(r)
    theta_vals = f0(grid)
    for k in range(n_steps):
        dW = rng.standard_normal(n_paths) * np.sqrt(dt)
        drift = theta_vals[k] - a * r[:, k]
        r[:, k + 1] = r[:, k] + drift * dt + sigma * dW
        A[:, k + 1] = A[:, k] + 0.5 * (r[:, k] + r[:, k + 1]) * dt
    return grid, r, A

def df_path(A, i, j):
    return np.exp(-(A[:, j] - A[:, i]))

def swap_value_hw(notional, K, grid, A_paths, pay_times, side="payer"):
    idx = np.searchsorted(grid, pay_times)
    n_paths, n_steps_plus = A_paths.shape
    values = np.zeros((n_paths, n_steps_plus))
    last = idx[-1] if len(idx) else n_steps_plus - 1
    for k in range(n_steps_plus):
        if len(idx) == 0:
            continue
        fut = [j for j in idx if j > k]
        if not fut:
            continue
        P_t_T = df_path(A_paths, k, last)
        pv_float = notional * (1 - P_t_T)
        pv_fixed = np.zeros(n_paths)
        prev = k
        for j in fut:
            tau = grid[j] - grid[prev]
            pv_fixed += notional * K * tau * df_path(A_paths, k, j)
            prev = j
        values[:, k] = (pv_float - pv_fixed) if side == "payer" else (pv_fixed - pv_float)
    return values

def cva_from_epe_grid(EPE, df0, S, grid, LGD):
    t = grid[1:]
    E = EPE[1:]
    D = np.array([df0(tt) for tt in t])
    Svals = S(t)
    dPD = np.diff(np.insert(Svals, 0, 1))
    CVA = LGD * np.sum(D * E * (-dPD))
    return CVA, t, E, D, -dPD, Svals

def norm_pdf(x): return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
def norm_cdf(x):
    # Erf-free approximation (Abramowitz–Stegun-like)
    p = 0.2316419
    b1,b2,b3,b4,b5 = 0.319381530,-0.356563782,1.781477937,-1.821255978,1.330274429
    x = np.asarray(x, float)
    t = 1.0 / (1.0 + p * np.abs(x))
    poly = (((((b5*t + b4)*t + b3)*t + b2)*t + b1)*t)
    phi = norm_pdf(x)
    cdf = 1.0 - phi * poly
    return np.where(x >= 0.0, cdf, 1.0 - cdf)

def build_collateral_paths(V_paths, grid, threshold, mta, margin_times=None, margin_freq_per_year=12):
    n_paths, n_steps = V_paths.shape
    C_paths = np.zeros_like(V_paths)
    if margin_times is not None and len(margin_times) > 0:
        T = grid[-1]
        mt = np.asarray(margin_times, float)
        mt = mt[(mt > 0) & (mt <= T + 1e-12)]
        margin_idx = np.unique(np.clip(np.searchsorted(grid, mt), 1, n_steps - 1)) if mt.size > 0 else np.array([], int)
    else:
        T = grid[-1]
        n_margins = int(np.floor(T * margin_freq_per_year + 1e-9))
        if n_margins <= 0:
            margin_idx = np.array([], int)
        else:
            mt = np.linspace(1.0 / margin_freq_per_year, n_margins / margin_freq_per_year, n_margins)
            margin_idx = np.unique(np.clip(np.searchsorted(grid, mt), 1, n_steps - 1))
    margin_set = set(margin_idx.tolist())
    C = np.zeros(n_paths)
    for k in range(n_steps):
        if k in margin_set:
            target = np.maximum(V_paths[:, k] - threshold, 0.0)
            delta = target - C
            C = np.where(np.abs(delta) >= mta, target, C)
            C = np.maximum(C, 0.0)
        C_paths[:, k] = C
    return C_paths

def heston_paths_euler(S0, v0, T, r, q, kappa, theta, xi, rho, steps, n_paths, seed=123):
    rng = np.random.default_rng(int(seed))
    dt = T / steps
    sqrt_dt = np.sqrt(dt)
    S = np.full((n_paths, steps + 1), float(S0))
    v = np.full((n_paths, steps + 1), float(v0))
    for k in range(steps):
        Z1 = rng.standard_normal(n_paths)
        Zp = rng.standard_normal(n_paths)
        Z2 = rho * Z1 + np.sqrt(1 - rho**2) * Zp
        vp = np.maximum(v[:, k], 0.0)
        v[:, k+1] = v[:, k] + kappa * (theta - vp) * dt + xi * np.sqrt(vp) * sqrt_dt * Z2
        vp_mid = np.maximum(0.5*(v[:, k] + v[:, k+1]), 0.0)
        drift = (r - q - 0.5 * vp_mid) * dt
        S[:, k+1] = S[:, k] * np.exp(drift + np.sqrt(vp_mid) * sqrt_dt * Z1)
    t = np.linspace(0.0, T, steps + 1)
    return t, S, v

def barrier_payoff_european(ST, K, B, opt_type, barrier_type):
    if opt_type == "Call":
        vanilla = np.maximum(ST - K, 0.0)
    else:
        vanilla = np.maximum(K - ST, 0.0)
    bt = str(barrier_type).lower()
    is_up = bt.startswith("up")
    is_in = bt.endswith("in")
    hit = (ST >= B) if is_up else (ST <= B)
    payoff = np.where(hit, vanilla, 0.0) if is_in else np.where(hit, 0.0, vanilla)
    return payoff, hit

# -----------------------------
# Initialize session defaults
# -----------------------------
if "swap" not in st.session_state:
    # sensible defaults (EUR)
    st.session_state.swap = {
        "ccy": "EUR",
        "notional": 10_000_000.0,
        "maturity": 10.0,
        "fixed_freq": CCY_PRESETS["EUR"]["fixed_freq"],
        "fixed_dcc": CCY_PRESETS["EUR"]["fixed_dcc"],
        "manual_K": 0.03,
        "curve_mode": "Flat r0",
        "zero_times": [0.0, 10.0],
        "zero_rates": [CCY_PRESETS["EUR"]["r0"], CCY_PRESETS["EUR"]["r0"]],
        "par_rate": np.nan,
    }

# -----------------------------
# Tabs (inputs separated)
# -----------------------------
tab_swap, tab_cva, tab_bar = st.tabs([
    "1️⃣ Swap (Par Rate) — Inputs + Result",
    "2️⃣ CVA & Exposure (CSA) — Inputs + Result",
    "3️⃣ Barrier Option (Heston, European)"
])

# === TAB 1: SWAP (all swap inputs here) ===
with tab_swap:
    st.subheader("Swap Inputs")
    colA, colB = st.columns(2)

    with colA:
        ccy = st.selectbox("Currency", list(CCY_PRESETS.keys()),
                           index=list(CCY_PRESETS.keys()).index(st.session_state.swap["ccy"]))
        p = CCY_PRESETS[ccy]
        notional = st.number_input(f"Notional ({ccy})", 1e5, 1e12, float(st.session_state.swap["notional"]), step=1e5, format="%.0f")
        maturity = st.number_input("Maturity (years)", 0.5, 60.0, float(st.session_state.swap["maturity"]), step=0.5)
        fixed_freq = st.selectbox("Fixed payments per year", [1,2,4], index=[1,2,4].index(st.session_state.swap["fixed_freq"]))
        fixed_dcc = st.selectbox("Fixed day count", ["30/360 US","30E/360","ACT/365F"],
                                 index=["30/360 US","30E/360","ACT/365F"].index(st.session_state.swap["fixed_dcc"]))
        manual_K = st.number_input("Manual fixed rate K (decimal)", 0.0, 1.0, float(st.session_state.swap["manual_K"]), step=0.0005, format="%.4f")

    with colB:
        st.markdown("**Initial Zero Curve**")
        curve_mode = st.radio("Curve input", ["Flat r0", "Custom zero curve"],
                              index=0 if st.session_state.swap["curve_mode"]=="Flat r0" else 1)
        if curve_mode == "Flat r0":
            r0_default = CCY_PRESETS[ccy]["r0"]
            r0 = st.number_input("Flat short rate r0", -1.0, 1.0, float(r0_default), step=0.0005, format="%.4f")
            zero_times = [0.0, maturity]
            zero_rates = [r0, r0]
        else:
            zt = st.text_input("Zero tenors (years)", ",".join(map(str, st.session_state.swap["zero_times"])))
            zr = st.text_input("Zero rates (decimals)", ",".join(map(lambda x:f"{x:g}", st.session_state.swap["zero_rates"])))
            try:
                zero_times = [float(x.strip()) for x in zt.split(",") if x.strip()]
                zero_rates = [float(x.strip()) for x in zr.split(",") if x.strip()]
            except Exception:
                st.error("Parse error in zero curve lists.")
                zero_times, zero_rates = [0.0, maturity], [CCY_PRESETS[ccy]["r0"], CCY_PRESETS[ccy]["r0"]]
            if len(zero_times) != len(zero_rates) or len(zero_times) < 2:
                st.error("Zero curve lists must have same length (≥2).")

    do_price = st.button("Compute Par Rate")

    # Compute par rate (and store to session)
    if do_price:
        f0 = forward_from_zero(zero_times, zero_rates)
        df0 = df0_from_zero(zero_times, zero_rates)
        dt = 1.0 / fixed_freq
        n_pay = int(np.round(maturity * fixed_freq))
        pay_times = np.linspace(dt, n_pay*dt, n_pay) if n_pay>0 else np.array([])
        D = np.array([df0(t) for t in pay_times]) if len(pay_times) else np.array([])
        taus = np.diff(np.insert(pay_times, 0, 0.0)) if len(pay_times) else np.array([])
        PV01 = float(np.sum(taus * D)) if len(pay_times) else 0.0
        P0T = float(df0(maturity))
        par_rate = (1.0 - P0T) / PV01 if PV01 > 0 else np.nan

        st.session_state.swap = {
            "ccy": ccy, "notional": notional, "maturity": maturity,
            "fixed_freq": fixed_freq, "fixed_dcc": fixed_dcc, "manual_K": manual_K,
            "curve_mode": curve_mode, "zero_times": zero_times, "zero_rates": zero_rates,
            "par_rate": par_rate
        }

        # Results
        st.subheader("Swap Result")
        c1, c2, c3 = st.columns(3)
        c1.metric("Par rate K*", f"{par_rate:.6%}" if np.isfinite(par_rate) else "NA")
        c2.metric("PV01", f"{PV01:,.6f}")
        c3.metric("P(0,T)", f"{P0T:,.6f}")
        if PV01 > 0:
            payer_npvn = (1.0 - P0T) - manual_K * PV01
            st.metric("Payer NPV / Notional (manual K)", f"{payer_npvn:,.6f}")
            st.line_chart(pd.DataFrame({"t": pay_times, "DF": D}).set_index("t"))
        else:
            st.warning("No fixed payment dates with this (T, freq).")

    with st.expander("Session snapshot (for CVA reuse)"):
        st.json(st.session_state.swap)

# === TAB 2: CVA & Exposure (all CVA inputs here) ===
with tab_cva:
    st.subheader("CVA Inputs")

    # Pull swap state
    sw = st.session_state.swap
    ccy = sw["ccy"]; maturity = float(sw["maturity"]); fixed_freq = int(sw["fixed_freq"])
    notional = float(sw["notional"]); K_used = float(sw["par_rate"]) if np.isfinite(sw["par_rate"]) else float(sw["manual_K"])
    zero_times, zero_rates = sw["zero_times"], sw["zero_rates"]

    # Show the inherited swap setup
    st.info(f"Using swap from Tab 1 — CCY: **{ccy}**, Notional: **{notional:,.0f}**, "
            f"T: **{maturity:.2f}y**, Fixed freq: **{fixed_freq}x**, "
            f"Fixed rate **K = {K_used:.6%}** (par if available), curve from Tab 1.")

    colL, colR = st.columns(2)

    with colL:
        st.markdown("**Hull–White (1F)**")
        p = CCY_PRESETS[ccy]
        a = st.number_input("Mean reversion a", 0.0001, 2.0, float(p["a"]), step=0.005, format="%.4f")
        sigma = st.number_input("Volatility σ", 0.0001, 2.0, float(p["sigma"]), step=0.0005, format="%.4f")

        st.markdown("**Credit**")
        credit_mode = st.radio("Credit input", ["Constant hazard λ", "CDS bootstrap (placeholder)"], index=0)
        if credit_mode == "Constant hazard λ":
            hazard = st.number_input("Hazard rate λ (annual)", 0.0001, 1.0, float(p["hazard"]), step=0.0005)
            LGD = st.number_input("LGD", 0.0, 1.0, float(p["LGD"]), step=0.05)
        else:
            LGD = st.number_input("LGD", 0.0, 1.0, 0.60, step=0.05)
            st.caption("CDS bootstrap omitted here; a flat proxy will be used.")

    with colR:
        st.markdown("**CSA (Collateral)**")
        use_csa = st.checkbox("Enable CSA", value=True)
        csa_threshold = st.number_input(f"Threshold ({ccy})", 0.0, 1e12, 0.0, step=10000.0, format="%.2f")
        csa_mta = st.number_input(f"Minimum Transfer Amount ({ccy})", 0.0, 1e12, 100000.0, step=10000.0, format="%.2f")
        margin_mode = st.radio("Margin calendar input", ["Frequency", "Custom times (years)"], index=0)
        if margin_mode == "Frequency":
            margin_label = st.selectbox(
                "Margin frequency",
                ["Daily (252/y)","Weekly (52/y)","Twice monthly (24/y)","Monthly (12/y)",
                 "Quarterly (4/y)","Semiannual (2/y)","Annual (1/y)"],
                index=3
            )
            freq_map = {"Daily (252/y)":252, "Weekly (52/y)":52, "Twice monthly (24/y)":24,
                        "Monthly (12/y)":12, "Quarterly (4/y)":4, "Semiannual (2/y)":2, "Annual (1/y)":1}
            margin_freq_per_year = freq_map[margin_label]
            custom_margin_times = None
        else:
            custom_str = st.text_input("Margin times (years, comma-separated)", "0.083333,0.25,0.5,0.75,1.0")
            try:
                custom_margin_times = sorted({float(x.strip()) for x in custom_str.split(",") if x.strip()})
            except Exception:
                custom_margin_times = []
            margin_freq_per_year = None

        st.markdown("**Simulation**")
        n_paths = st.number_input("Monte Carlo paths", 1000, 200000, 50000, step=1000)
        seed = st.number_input("Random seed", 0, 10**9, 123)

    run_cva = st.button("Run CVA")

    if run_cva:
        # Build curve and schedules from Tab 1
        f0 = forward_from_zero(zero_times, zero_rates)
        df0 = df0_from_zero(zero_times, zero_rates)
        dt = 1.0 / fixed_freq
        n_pay = int(np.round(maturity * fixed_freq))
        pay_times = np.linspace(dt, n_pay*dt, n_pay) if n_pay>0 else np.array([])

        # Simulate HW
        grid, r_paths, A_paths = simulate_hw_paths(a, sigma, f0, maturity, dt, int(n_paths), seed=int(seed))
        side_key = "payer" if True else "receiver"  # exposure is positive when we're ITM; sign handled in swap_value_hw

        # MtM paths at par K (or manual if par not computed)
        V_paths = swap_value_hw(notional, K_used, grid, A_paths, pay_times, side_key)

        # CSA
        if use_csa:
            C_paths = build_collateral_paths(
                V_paths, grid,
                threshold=csa_threshold, mta=csa_mta,
                margin_times=custom_margin_times if margin_mode=="Custom times (years)" else None,
                margin_freq_per_year=margin_freq_per_year if margin_mode=="Frequency" else 12
            )
            V_eff_paths = V_paths - C_paths
        else:
            V_eff_paths = V_paths

        # EPE (MC)
        exposures_mc_paths = np.maximum(V_eff_paths, 0.0)
        EPE_mc = exposures_mc_paths.mean(axis=0)

        # EPE (Moment Matching) on V_eff
        mu = V_eff_paths.mean(axis=0)
        s  = V_eff_paths.std(axis=0, ddof=1)
        k  = mu / np.minimum(np.maximum(s, 1e-12), 1e12)
        Phi = norm_cdf(k); phi = norm_pdf(k)
        EPE_mm = np.where(s < 1e-14, np.maximum(mu, 0.0), s * phi + mu * Phi)

        # Credit survival
        if credit_mode == "Constant hazard λ":
            S = lambda t: np.exp(-hazard * np.atleast_1d(t))
        else:
            flat_lambda = 0.02
            S = lambda t: np.exp(-flat_lambda * np.atleast_1d(t))

        # CVA from both methods
        CVA_mc, t_epe, E_mc, D0t, dPD, Svals = cva_from_epe_grid(EPE_mc, df0, S, grid, LGD)
        CVA_mm, _, E_mm, _, _, _ = cva_from_epe_grid(EPE_mm, df0, S, grid, LGD)

        st.subheader("CVA Results")
        c1, c2, c3 = st.columns(3)
        c1.metric("CVA (Monte Carlo)", f"{CVA_mc:,.2f}")
        c2.metric("CVA (Moment Matching)", f"{CVA_mm:,.2f}")
        diff = CVA_mm - CVA_mc
        rel = (diff / CVA_mc * 100 if CVA_mc else np.nan)
        c3.metric("Difference", f"{diff:,.2f}", f"{rel:+.2f}%")

        st.markdown("**EPE comparison (MC vs Moment-Matching)**")
        st.line_chart(pd.DataFrame({"t": grid, "EPE (MC)": EPE_mc, "EPE (MM)": EPE_mm}).set_index("t"))

        st.markdown("**Default increment dPD(t_k)**")
        st.area_chart(pd.DataFrame({"t": t_epe, "dPD": dPD}).set_index("t"))

        out = pd.DataFrame({
            "t": t_epe,
            "EPE_MC": E_mc,
            "EPE_MM": np.interp(t_epe, grid[1:], EPE_mm[1:]),
            "dPD": dPD,
            "DF": D0t
        })
        st.dataframe(out, use_container_width=True)
        st.download_button("⬇️ Download CVA data", out.to_csv(index=False),
                           file_name=f"cva_compare_{ccy}_CSA.csv")

        if use_csa:
            C_mean = C_paths.mean(axis=0)
            st.subheader("CSA Diagnostics")
            st.line_chart(pd.DataFrame({"t": grid, "Mean Collateral": C_mean}).set_index("t"))

# === TAB 3: Barrier (Heston, European) — unchanged logic ===
with tab_bar:
    st.subheader("Barrier Option — Heston MC (European barrier only)")
    colL, colR = st.columns(2)
    with colL:
        S0 = st.number_input("Spot S₀", 0.0, 1e9, 100.0, step=1.0)
        K_opt = st.number_input("Strike K", 0.0, 1e9, 100.0, step=1.0)
        T_opt = st.number_input("Maturity T (years)", 0.01, 50.0, 2.0, step=0.01)
        q_div = st.number_input("Dividend yield q (decimal)", 0.0, 1.0, 0.00, step=0.001, format="%.4f")
        opt_type = st.selectbox("Option type", ["Call", "Put"], index=0)
    with colR:
        barrier_type = st.selectbox("Barrier type", ["Up-and-Out", "Down-and-Out", "Up-and-In", "Down-and-In"], index=0)
        B = st.number_input("Barrier level B", 0.0, 1e9, 120.0, step=1.0)
        steps_per_year = st.number_input("Time steps per year (SDE integration)", 12, 5000, 252, step=1)
        n_paths_bar = st.number_input("MC paths (barrier)", 1000, 500000, 20000, step=1000)

    go_bar = st.button("Price Barrier Option")

    if go_bar:
        # Use curve from Swap tab; fallback if not set
        zero_times = st.session_state.swap["zero_times"]; zero_rates = st.session_state.swap["zero_rates"]
        df0 = df0_from_zero(zero_times, zero_rates)
        DF0T = float(df0(T_opt))
        r_eff = -np.log(max(DF0T, 1e-300)) / T_opt

        steps = int(np.ceil(T_opt * steps_per_year))
        t_grid, S_paths, v_paths = heston_paths_euler(
            S0, 0.04, T_opt, r_eff, q_div, 1.5, 0.04, 0.5, -0.7,
            steps=steps, n_paths=int(n_paths_bar), seed=123
        )
        ST = S_paths[:, -1]
        payoff_paths, hit = barrier_payoff_european(ST, K_opt, B, opt_type, barrier_type)
        price = DF0T * payoff_paths.mean()
        stderr = DF0T * payoff_paths.std(ddof=1) / np.sqrt(len(payoff_paths))
        hit_rate = float(hit.mean())

        st.subheader("Barrier Price (Heston MC, European)")
        c1, c2, c3 = st.columns(3)
        c1.metric("Price", f"{price:,.6f}")
        c2.metric("Std. Error", f"{stderr:,.6f}")
        c3.metric("Hit rate", f"{hit_rate:.2%}")
        st.caption(f"DF(0,T)={DF0T:.6f} | r_eff={r_eff:.4%} | Steps={steps} (~{steps_per_year}/y)")

        # Sample paths
        take = min(100, S_paths.shape[0])
        df_paths = pd.DataFrame({"t": t_grid})
        for i in range(take):
            df_paths[f"path_{i+1}"] = S_paths[i]
        st.line_chart(df_paths.set_index("t"))

        # Payoff vs S(T)
        ST_axis = np.linspace(0.0, max(1.2 * max(float(ST.max()), float(B)), 1e-6), 400)
        vanilla_curve = np.maximum(ST_axis - K_opt, 0.0) if opt_type == "Call" else np.maximum(K_opt - ST_axis, 0.0)

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.scatter(ST, payoff_paths, s=6, alpha=0.35)
        ax.plot(ST_axis, vanilla_curve, linewidth=2)
        ax.axvline(B, linestyle="--")
        ax.set_xlabel("S(T)"); ax.set_ylabel("Payoff")
        ax.set_title(f"{opt_type} – {barrier_type} (European)")
        st.pyplot(fig)

        out_bar = pd.DataFrame({"ST": ST, "Payoff": payoff_paths, "HitBarrier": hit.astype(int)})
        st.dataframe(out_bar.head(2000), use_container_width=True)
        st.download_button("⬇️ Download path payoffs (CSV)", out_bar.to_csv(index=False),
                           file_name=f"barrier_paths_{opt_type}_{barrier_type}_EURO.csv")
