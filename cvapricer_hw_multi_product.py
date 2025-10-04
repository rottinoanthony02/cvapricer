# cvapricer_hw_multi_product.py
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="CVA – IRS & EURUSD CCS (Hull–White 1F + CDS bootstrap)", layout="wide")
st.title("💳 CVA – IRS & EURUSD Cross-Currency Swap (HW1F rates, CDS hazard bootstrap)")

# ========= Presets (conventions + baseline params) =========
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

# ========= Sidebar: Product selection =========
with st.sidebar:
    st.header("Product")
    product = st.selectbox("Type", ["IRS (single currency)", "EURUSD Cross-Currency Swap"], index=0)

# ========= Common helpers =========
def parse_csv_floats(s): return [float(x.strip()) for x in s.split(",") if x.strip()]

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
        zt = z(t); zp = (z(t+h)-z(t-h))/(2*h)
        return float(zt + t*zp)
    return f0

def df0_from_zero(times, zeros):
    z = interp_zero(times, zeros)
    return lambda t: float(np.exp(-z(t)*t))

def theta_from_f0_grid(grid, a, sigma, f0):
    h = 1e-4
    dfdt = (np.array([f0(t+h) for t in grid]) - np.array([f0(t-h) for t in grid]))/(2*h)
    return dfdt + a*np.array([f0(t) for t in grid]) + (sigma**2/(2*a))*(1.0 - np.exp(-2*a*grid))

def simulate_hw_paths(a, sigma, f0, T, dt, n_paths, seed=123):
    n_steps = int(np.ceil(T/dt))
    grid = np.linspace(0.0, n_steps*dt, n_steps+1)
    rng = np.random.default_rng(int(seed))
    r = np.zeros((n_paths, n_steps+1)); r[:,0] = f0(1e-6)
    A = np.zeros_like(r)
    theta = theta_from_f0_grid(grid, a, sigma, f0)
    for k in range(n_steps):
        dW = rng.standard_normal(n_paths)*np.sqrt(dt)
        drift = theta[k] - a*r[:,k]
        r[:,k+1] = r[:,k] + drift*dt + sigma*dW
        A[:,k+1] = A[:,k] + 0.5*(r[:,k] + r[:,k+1])*dt
    return grid, r, A

def df_path(A, idx_from, idx_to):
    return np.exp(-(A[:, idx_to] - A[:, idx_from]))

# ---- CDS bootstrap (piecewise hazards) ----
def cds_pay_grid(T, pay_freq=4):
    n = int(np.round(T * pay_freq))
    return np.linspace(0.0, T, n+1)

def cds_leg_rpv01(times_pay, df0, S):
    acc = 0.0
    for i in range(1, len(times_pay)):
        tau = times_pay[i] - times_pay[i-1]; t = times_pay[i]
        acc += tau * df0(t) * S(t)
    return acc

def cds_leg_protection(times_pay, df0, S, R):
    acc = 0.0
    for i in range(1, len(times_pay)):
        t0, t1 = times_pay[i-1], times_pay[i]
        acc += df0(t1) * (S(t0) - S(t1)) * (1.0 - R)
    return acc

def bootstrap_piecewise_hazard(tenors, spreads_bps, df0, R=0.4, pay_freq=4, max_iter=100, tol=1e-10):
    tenors = np.array(tenors, float); spreads = np.array(spreads_bps, float) * 1e-4
    cut = np.concatenate([[0.0], tenors]); hazards = []
    def S_with(hs):
        def S(t):
            t = np.atleast_1d(t); Svals = np.ones_like(t, float)
            for j in range(1, len(cut)):
                lam = hs[j-1] if j-1 < len(hs) else (hs[-1] if hs else 0.0)
                t0,t1 = cut[j-1], cut[j]
                dt = np.clip(np.minimum(t, t1) - t0, 0, None)
                Svals *= np.exp(-lam * dt)
            if hs:
                extra = np.clip(t - cut[-1], 0, None)
                Svals *= np.exp(-hs[-1] * extra)
            return Svals if Svals.shape != () else float(Svals)
        return S
    for i, T in enumerate(tenors):
        grid = cds_pay_grid(T, 4); s = spreads[i]
        def f(lmb):
            S_try = S_with(hazards + [lmb])
            return s * cds_leg_rpv01(grid, df0, S_try) - cds_leg_protection(grid, df0, S_try, R)
        lo, hi = 1e-8, 5.0; flo = f(lo); mid = None
        for _ in range(max_iter):
            mid = 0.5*(lo+hi); fm = f(mid)
            if abs(fm) < tol: hazards.append(mid); break
            if np.sign(fm) == np.sign(flo): lo, flo = mid, fm
            else: hi = mid
        else: hazards.append(mid if mid is not None else lo)
    return hazards, S_with(hazards)

def cva_from_epe_grid(EPE, df0, S, grid, LGD):
    t = grid[1:]; E = EPE[1:]
    D = np.array([df0(tt) for tt in t]); S_vals = S(t)
    dPD = np.empty_like(t); dPD[0] = 1.0 - S_vals[0]; dPD[1:] = S_vals[:-1] - S_vals[1:]
    return LGD * float(np.sum(D * E * dPD)), t, E, D, dPD, S_vals

# ========= IRS branch (reuse HW1F single-curve) =========
def grid_indices_for_payments(grid, pay_times):
    idx = np.searchsorted(grid, pay_times, side="left")
    return np.clip(idx, 1, len(grid)-1).tolist()

def swap_exposure_hw_irs(notional, K, grid, A_paths, pay_times, side="payer"):
    n_paths, n_steps_plus = A_paths.shape
    exposures = np.zeros((n_paths, n_steps_plus))
    pay_idx = grid_indices_for_payments(grid, pay_times); last_idx = pay_idx[-1]
    for k in range(n_steps_plus):
        fut = [j for j in pay_idx if j > k]
        if not fut: continue
        P_t_T = df_path(A_paths, k, last_idx)                  # float approx
        pv_float = notional * (1.0 - P_t_T)
        pv_fixed = np.zeros(n_paths); prev = k
        for j in fut:
            tau = grid[j] - grid[prev]; df = df_path(A_paths, k, j)
            pv_fixed += notional * K * tau * df; prev = j
        val = (pv_float - pv_fixed) if side=="payer" else (pv_fixed - pv_float)
        exposures[:, k] = np.maximum(val, 0.0)
    return exposures

# ========= CCS branch (HW1F USD & EUR + FX GBM with correlations) =========
def simulate_hw_fx_paths(a_usd, sig_usd, f0_usd,
                         a_eur, sig_eur, f0_eur,
                         sig_fx, S0, T, dt, n_paths,
                         rho_usd_eur=0.3, rho_usd_fx=0.2, rho_eur_fx=0.1, seed=123):
    n_steps = int(np.ceil(T/dt))
    grid = np.linspace(0.0, n_steps*dt, n_steps+1)
    rng = np.random.default_rng(int(seed))

    r_usd = np.zeros((n_paths, n_steps+1)); r_usd[:,0] = f0_usd(1e-6)
    r_eur = np.zeros((n_paths, n_steps+1)); r_eur[:,0] = f0_eur(1e-6)
    A_usd = np.zeros_like(r_usd); A_eur = np.zeros_like(r_eur)
    FX = np.zeros((n_paths, n_steps+1)); FX[:,0] = S0

    theta_usd = theta_from_f0_grid(grid, a_usd, sig_usd, f0_usd)
    theta_eur = theta_from_f0_grid(grid, a_eur, sig_eur, f0_eur)

    # Correlation matrix & Cholesky
    R = np.array([[1.0,         rho_usd_eur, rho_usd_fx],
                  [rho_usd_eur, 1.0,         rho_eur_fx],
                  [rho_usd_fx,  rho_eur_fx,  1.0       ]], float)
    # Ensure PSD
    try:
        L = np.linalg.cholesky(R)
    except np.linalg.LinAlgError:
        # small diagonal bump if needed
        w, v = np.linalg.eigh(R)
        w = np.clip(w, 1e-10, None)
        R_pd = (v * w) @ v.T
        L = np.linalg.cholesky(R_pd)

    sdt = np.sqrt(dt)
    for k in range(n_steps):
        Z = rng.standard_normal((n_paths, 3))
        E = Z @ L.T
        dW_usd, dW_eur, dW_fx = E[:,0]*sdt, E[:,1]*sdt, E[:,2]*sdt

        # USD/EUR short rates
        drift_u = theta_usd[k] - a_usd * r_usd[:,k]
        drift_e = theta_eur[k] - a_eur * r_eur[:,k]
        r_usd[:,k+1] = r_usd[:,k] + drift_u*dt + sig_usd * dW_usd
        r_eur[:,k+1] = r_eur[:,k] + drift_e*dt + sig_eur * dW_eur
        A_usd[:,k+1] = A_usd[:,k] + 0.5*(r_usd[:,k] + r_usd[:,k+1])*dt
        A_eur[:,k+1] = A_eur[:,k] + 0.5*(r_eur[:,k] + r_eur[:,k+1])*dt

        # FX GBM under USD measure: dln S = ((r_USD - r_EUR) - 0.5σ_fx^2)dt + σ_fx dW_fx
        mu = (r_usd[:,k] - r_eur[:,k]) - 0.5*sig_fx**2
        FX[:,k+1] = FX[:,k] * np.exp(mu*dt + sig_fx * dW_fx)

    return grid, r_usd, A_usd, r_eur, A_eur, FX

def swap_exposure_ccs_hw(N_usd, N_eur, s_usd, s_eur,
                         grid, A_usd, A_eur, FX,
                         pay_times_usd, pay_times_eur,
                         side="Pay USD, Receive EUR"):
    """
    Float–float CCS with constant notionals; spreads s_usd/s_eur in decimal per annum.
    Exposure is the positive part of the net USD value at each grid time, including
    final notional re-exchange at maturity.
    """
    n_paths, n_steps_plus = A_usd.shape
    exposures = np.zeros((n_paths, n_steps_plus))

    idx_usd = grid_indices_for_payments(grid, pay_times_usd)
    idx_eur = grid_indices_for_payments(grid, pay_times_eur)
    last_usd, last_eur = idx_usd[-1], idx_eur[-1]

    sign_usd_leg = -1 if side.lower().startswith("pay usd") else +1
    sign_eur_leg = +1 if side.lower().endswith("receive eur") else -1  # "Pay EUR, Receive USD" => -1

    for k in range(n_steps_plus):
        fut_usd = [j for j in idx_usd if j > k]
        fut_eur = [j for j in idx_eur if j > k]
        if not (fut_usd or fut_eur): continue

        # USD leg (float approx + spread)
        P_u_T = df_path(A_usd, k, last_usd)
        pv_usd_float = N_usd * (1.0 - P_u_T)
        pv_usd_spread = np.zeros(n_paths); prev = k
        for j in fut_usd:
            tau = grid[j] - grid[prev]; df = df_path(A_usd, k, j)
            pv_usd_spread += N_usd * s_usd * tau * df; prev = j
        pv_usd_leg = pv_usd_float + pv_usd_spread

        # EUR leg (value in EUR, then convert to USD at FX_k)
        P_e_T = df_path(A_eur, k, last_eur)
        pv_eur_float = N_eur * (1.0 - P_e_T)
        pv_eur_spread = np.zeros(n_paths); prev = k
        for j in fut_eur:
            tau = grid[j] - grid[prev]; df = df_path(A_eur, k, j)
            pv_eur_spread += N_eur * s_eur * tau * df; prev = j
        pv_eur_leg_eur = pv_eur_float + pv_eur_spread
        FX_k = FX[:,k]
        pv_eur_leg_usd = FX_k * pv_eur_leg_eur

        # Final notional re-exchange at maturity
        pv_not_usd = N_usd * df_path(A_usd, k, last_usd)
        pv_not_eur_usd = FX_k * N_eur * df_path(A_eur, k, last_eur)

        # Net value (USD)
        if side.lower().startswith("pay usd"):
            value = (+ pv_eur_leg_usd + pv_not_eur_usd) - (pv_usd_leg + pv_not_usd)
        else:  # Pay EUR, Receive USD
            value = (+ pv_usd_leg + pv_not_usd) - (pv_eur_leg_usd + pv_not_eur_usd)

        exposures[:, k] = np.maximum(value, 0.0)

    return exposures

# ========= UI branches =========
if product == "IRS (single currency)":
    with st.sidebar:
        st.header("IRS setup")
        ccy = st.selectbox("Currency", list(CCY_PRESETS.keys()), index=0)
        p = CCY_PRESETS[ccy]; sym = p["symbol"]
        side = st.selectbox("Swap side", ["Payer fixed", "Receiver fixed"], index=0)
        notional = st.number_input(f"Notional ({ccy})", 1e5, 1e12, 10_000_000.0, step=1e5, format="%.0f")
        maturity = st.number_input("Maturity (years)", 1.0, 50.0, 10.0, step=0.5)
        fixed_freq = st.selectbox("Fixed payments per year", [1,2,4], index=[1,2,4].index(p["fixed_freq"]))
        fixed_rate = st.number_input("Fixed rate K (decimal)", 0.0, 1.0, 0.03, step=0.0005, format="%.4f")
        st.caption(f"Accrual τ ≈ 1/freq. Day count label: {p['fixed_dcc']}")

        st.header("Initial curve (for HW fit)")
        curve_mode = st.radio("Curve input", ["Flat r₀"]()_
