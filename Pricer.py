# cvapricer_multi_ccy_cva_cds_hw.py
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="CVA Multi-CCY", layout="wide")
st.title("CVA PRICER by A.Rottino")

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
# Sidebar parameters
# -----------------------------
with st.sidebar:
    st.header("Instrument")
    ccy = st.selectbox("Currency", list(CCY_PRESETS.keys()), index=0)
    p = CCY_PRESETS[ccy]
    sym = p["symbol"]

    side = st.selectbox("Swap side", ["Payer fixed", "Receiver fixed"], index=0)
    notional = st.number_input(f"Notional ({ccy})", 1e5, 1e12, 10_000_000.0, step=1e5, format="%.0f")
    maturity = st.number_input("Maturity (years)", 1.0, 50.0, 10.0, step=0.5)
    fixed_freq = st.selectbox("Fixed payments per year", [1, 2, 4], index=[1,2,4].index(p["fixed_freq"]))
    fixed_dcc = st.selectbox("Fixed day count", ["30/360 US", "30E/360", "ACT/365F"],
                             index=["30/360 US","30E/360","ACT/365F"].index(p["fixed_dcc"]))
    fixed_rate = st.number_input("Manual fixed rate K (decimal)", 0.0, 1.0, 0.03, step=0.0005, format="%.4f")

    st.caption("Accrual τ ≈ 1/frequency.")

    st.header("Curve input")
    curve_mode = st.radio("Zero curve mode", ["Flat r₀", "Custom zero curve"], index=0)
    if curve_mode == "Flat r₀":
        r0 = st.number_input("Flat short rate r₀", -1.0, 1.0, float(p["r0"]), step=0.0005, format="%.4f")
        zero_times, zero_rates = [0.0, maturity], [r0, r0]
    else:
        st.caption("Comma-separated tenors and rates.")
        zero_times = [float(x) for x in st.text_input("Tenors (years)", "0,1,2,5,10,30").split(",")]
        zero_rates = [float(x) for x in st.text_input("Zero rates (decimals)", "0.02,0.022,0.023,0.024,0.025,0.026").split(",")]

    st.header("Hull–White (1F)")
    a = st.number_input("Mean reversion a", 0.0001, 2.0, float(p["a"]), step=0.005, format="%.4f")
    sigma = st.number_input("Volatility σ", 0.0001, 2.0, float(p["sigma"]), step=0.0005, format="%.4f")

    st.header("Credit")
    credit_mode = st.radio("Credit input", ["Constant hazard λ", "CDS bootstrap"], index=0)
    if credit_mode == "Constant hazard λ":
        hazard = st.number_input("Hazard rate λ (annual)", 0.0001, 1.0, float(p["hazard"]), step=0.0005)
        LGD = st.number_input("LGD", 0.0, 1.0, float(p["LGD"]), step=0.05)
        recovery = 1 - LGD
    else:
        LGD = st.number_input("LGD", 0.0, 1.0, 0.6, step=0.05)
        recovery = 1 - LGD
        cds_tenors = [float(x) for x in st.text_input("CDS tenors (years)", "1,3,5,7,10").split(",")]
        cds_spreads = [float(x) for x in st.text_input("CDS spreads (bps)", "80,100,120,140,160").split(",")]

    st.header("Simulation")
    n_paths = st.number_input("Monte Carlo paths", 1000, 200000, 50000, step=1000)
    seed = st.number_input("Random seed", 0, 10**9, 123)
    show_paths = st.slider("Show N rate paths", 1, 200, 50)

    compute_btn = st.button("Run CVA")

# -----------------------------
# Vectorized curve helpers (FIXED)
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

# -----------------------------
# Other helpers
# -----------------------------
def simulate_hw_paths(a, sigma, f0, T, dt, n_paths, seed=123):
    n_steps = int(np.ceil(T / dt))
    grid = np.linspace(0, n_steps * dt, n_steps + 1)
    rng = np.random.default_rng(seed)
    r = np.zeros((n_paths, n_steps + 1))
    A = np.zeros_like(r)
    theta_vals = f0(grid)
    for k in range(n_steps):
        dW = rng.standard_normal(n_paths) * np.sqrt(dt)
        drift = theta_vals[k] - a * r[:, k]
        r[:, k + 1] = r[:, k] + drift * dt + sigma * dW
        A[:, k + 1] = A[:, k] + 0.5 * (r[:, k] + r[:, k + 1]) * dt
    return grid, r, A

def df_path(A, i, j): return np.exp(-(A[:, j] - A[:, i]))

def swap_value_hw(notional, K, grid, A_paths, pay_times, side="payer"):
    idx = np.searchsorted(grid, pay_times)
    n_paths, n_steps = A_paths.shape
    values = np.zeros((n_paths, n_steps))
    last = idx[-1]
    for k in range(n_steps):
        fut = [j for j in idx if j > k]
        if not fut: continue
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
    p = 0.2316419
    b1,b2,b3,b4,b5 = 0.319381530,-0.356563782,1.781477937,-1.821255978,1.330274429
    t = 1 / (1 + p * np.abs(x))
    poly = (((((b5*t + b4)*t + b3)*t + b2)*t + b1)*t)
    phi = norm_pdf(x)
    cdf = 1 - phi * poly
    return np.where(x >= 0, cdf, 1 - cdf)

# -----------------------------
# Build curve + tabs
# -----------------------------
f0 = forward_from_zero(zero_times, zero_rates)
df0 = df0_from_zero(zero_times, zero_rates)
dt = 1 / fixed_freq
pay_times = np.linspace(dt, maturity, int(maturity * fixed_freq))

# Par rate calculation
D = np.array([df0(t) for t in pay_times])
taus = np.diff(np.insert(pay_times, 0, 0))
PV01 = np.sum(taus * D)
P0T = df0(maturity)
par_rate = (1 - P0T) / PV01 if PV01 > 0 else np.nan
st.session_state["par_rate"] = par_rate

# -----------------------------
# Tabs (Par Rate first)
# -----------------------------
tab_par, tab_cva = st.tabs(["1️⃣ Par Swap Rate", "2️⃣ CVA & Exposure"])

# --- PAR RATE TAB ---
with tab_par:
    st.subheader("Par Swap Rate")
    c1, c2, c3 = st.columns(3)
    c1.metric("Par rate K*", f"{par_rate:.6%}")
    c2.metric("PV01", f"{PV01:,.6f}")
    c3.metric("P(0,T)", f"{P0T:,.6f}")
    st.caption("Calculated from zero curve. This K* will be used automatically in the CVA pricer.")
    npv_per_notional_payer = (1 - P0T) - fixed_rate * PV01
    st.metric("Payer NPV (manual K)", f"{npv_per_notional_payer:,.6f}")
    st.line_chart(pd.DataFrame({"t": pay_times, "DF": D}).set_index("t"))

# --- CVA TAB ---
with tab_cva:
    K_used = st.session_state["par_rate"]
    st.info(f"Using **Par rate K* = {K_used:.6%}** from previous tab.")
    if compute_btn:
        grid, r_paths, A_paths = simulate_hw_paths(a, sigma, f0, maturity, dt, n_paths, seed)
        side_key = "payer" if side.lower().startswith("payer") else "receiver"
        V_paths = swap_value_hw(notional, K_used, grid, A_paths, pay_times, side_key)
        exposures = np.maximum(V_paths, 0)
        EPE_mc = exposures.mean(axis=0)

        mu = V_paths.mean(axis=0)
        s = V_paths.std(axis=0, ddof=1)
        k = mu / np.maximum(s, 1e-12)
        Phi, phi = norm_cdf(k), norm_pdf(k)
        EPE_mm = s * phi + mu * Phi
        EPE_mm = np.where(s < 1e-14, np.maximum(mu, 0), EPE_mm)

        S = (lambda t: np.exp(-hazard * np.atleast_1d(t))) if credit_mode == "Constant hazard λ" else (lambda t: np.exp(-0.02 * np.atleast_1d(t)))
        CVA_mc, t_epe_mc, E_mc, D_mc, dPD_mc, Svals = cva_from_epe_grid(EPE_mc, df0, S, grid, LGD)
        CVA_mm, _, E_mm, _, _, _ = cva_from_epe_grid(EPE_mm, df0, S, grid, LGD)

        st.subheader("CVA Results")
        c1, c2, c3 = st.columns(3)
        c1.metric("CVA (Monte Carlo)", f"{CVA_mc:,.2f}")
        c2.metric("CVA (Moment Matching)", f"{CVA_mm:,.2f}")
        diff = CVA_mm - CVA_mc
        rel = (diff / CVA_mc * 100 if CVA_mc else np.nan)
        c3.metric("Difference", f"{diff:,.2f}", f"{rel:+.2f}%")

        st.line_chart(pd.DataFrame({"t": grid, "EPE (MC)": EPE_mc, "EPE (MM)": EPE_mm}).set_index("t"))
        st.area_chart(pd.DataFrame({"t": t_epe_mc, "dPD": dPD_mc}).set_index("t"))

        out = pd.DataFrame({"t": t_epe_mc, "EPE_MC": E_mc, "EPE_MM": np.interp(t_epe_mc, grid[1:], EPE_mm[1:]), "dPD": dPD_mc, "DF": D_mc})
        st.dataframe(out, use_container_width=True)
        st.download_button("⬇️ Download CVA data", out.to_csv(index=False), file_name=f"cva_compare_{ccy}.csv")
