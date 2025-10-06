# pricer_v5.py
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Swap Par • CVA (CSA) • Heston Barrier (EU)", layout="wide")
st.title("Swap Par • CVA (CSA) • Heston Barrier (European) — by A. Rottino")

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
    if len(idx) == 0:
        return values
    last = idx[-1]
    for k in range(n_steps_plus):
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
    return CVA, t
