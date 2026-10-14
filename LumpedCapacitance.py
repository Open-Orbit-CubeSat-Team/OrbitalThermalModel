"""
lumped_comparison.py
Computes:
  1. Effective radiation h and Biot number over time
  2. Lumped capacitance temperature history
  3. Comparison plots: Bi(t), T_lumped vs T_FD_avg, Q_abs vs Q_emit
"""

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.time import Time
from OrbitHTModelFunctions import precompute_rad_table
import os

# ── CONFIGURATION — must match CubeSatFiniteDiff.py ───────────────────────────
H5_PATH    = "sim_outputs/thermal_results_s1.h5"
OUT_DIR    = "sim_outputs"

TLE_LINE1  = "1 25544U 98067A   19343.69339541  .00001764  00000-0  38792-4 0  9991"
TLE_LINE2  = "2 25544  51.6439 211.2001 0007417  17.6667  85.6398 15.50103472202482"
T0         = Time("2026-02-22T12:00:00", scale="utc")

# Geometry
x_span = y_span = z_span = 1e-1   # m
i_th   = j_th   = k_th   = 5e-3   # wall thickness m
dx = dy = dz = 1e-3                # m

A_DIR = np.array([dy*dz, dy*dz, dx*dz, dx*dz, dx*dy, dx*dy])  # 6 face areas

# Shell volume (full cube minus hollow interior)
V_shell = (x_span*y_span*z_span) - (
           (x_span - 2*i_th) * (y_span - 2*j_th) * (z_span - 2*k_th))

A_total = 6 * x_span * x_span   # total outer surface area [m^2]

# Material — Aluminum 6061-T6
rho_alum   = 2700.0    # kg/m^3
cp_alum    = 896.0     # J/kg/K
k_alum     = 167.0     # W/m/K
eps_alum   = 0.05
alpha_alum = 0.05
sigma      = 5.670e-8  # W/m^2/K^4

L_c        = i_th      # characteristic length = wall thickness [m]

# Orbital geometry
R_EARTH    = 6378.137  # km
ALT_ORBIT  = 400.0     # km  (ISS TLE ~ 400 km)
k_orbit    = R_EARTH / (R_EARTH + ALT_ORBIT)

# Deep space background temperature
T_space    = 3.0       # K

# Lumped mass
m = rho_alum * V_shell

# ── LOAD FD RESULTS ───────────────────────────────────────────────────────────
print("Loading HDF5 results...")
with h5py.File(H5_PATH, "r") as f:
    time_sec = f["time_sec"][:]
    avg_T_FD = f["avg_T"][:]

print(f"  {len(time_sec)} snapshots  |  "
      f"t = {time_sec[0]:.0f} – {time_sec[-1]:.0f} s")

# ── RADIATION TABLE ───────────────────────────────────────────────────────────
print("Precomputing radiation table...")
t_end = float(time_sec[-1])
rad_sample_times, rad_table = precompute_rad_table(
    TLE_LINE1, TLE_LINE2, T0, t_end, 20.0, k_orbit
)

# Q_abs per unit (alpha=1, A=1) summed over all 6 faces at each sample time
# rad_table shape: (n_samp, 6)
# Scale by alpha and actual face areas
Q_abs_samples = np.zeros(len(rad_sample_times))
for fi in range(6):
    Q_abs_samples += alpha_alum * A_DIR[fi] * rad_table[:, fi]

# Interpolate Q_abs onto FD snapshot times
Q_abs = np.interp(time_sec.astype(float), rad_sample_times, Q_abs_samples)

# ── BIOT NUMBER OVER TIME ─────────────────────────────────────────────────────
# Linearized effective radiation h:
#   h_rad = eps * sigma * (T^2 + T_space^2) * (T + T_space)
T_K     = avg_T_FD
h_rad   = eps_alum * sigma * (T_K**2 + T_space**2) * (T_K + T_space)
Bi      = h_rad * L_c / k_alum

print(f"  Bi range: {Bi.min():.4e} – {Bi.max():.4e}")
print(f"  Max Bi = {Bi.max():.4e}  ({'VALID' if Bi.max() < 0.1 else 'INVALID'} lumped assumption)")

# ── LUMPED CAPACITANCE ODE — forward Euler ────────────────────────────────────
# m*cp * dT/dt = Q_abs(t) - eps*sigma*A_total*T^4
print("Solving lumped capacitance ODE...")

n        = len(time_sec)
T_lumped = np.zeros(n)
T_lumped[0] = avg_T_FD[0]   # same initial condition as FD

for i in range(n - 1):
    dt_i   = time_sec[i+1] - time_sec[i]
    T_i    = T_lumped[i]
    Q_emit = eps_alum * sigma * A_total * T_i**4
    dTdt   = (Q_abs[i] - Q_emit) / (m * cp_alum)
    T_lumped[i+1] = T_i + dTdt * dt_i

# ── EMISSION HISTORY (using FD avg T as proxy) ───────────────────────────────
Q_emit_FD = eps_alum * sigma * A_total * avg_T_FD**4

# ── PLOTS ─────────────────────────────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)
t_min = time_sec / 60.0   # convert to minutes

# --- Plot 1: Biot Number vs Time ---
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(t_min, Bi, color="tab:purple", linewidth=1.2)
ax.axhline(0.1, color="red", linestyle="--", linewidth=1.0, label="Bi = 0.1 threshold")
ax.set_xlabel("Time [min]")
ax.set_ylabel("Biot Number [-]")
ax.set_title("Effective Radiation Biot Number vs Time")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_yscale("log")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "biot_number.png"), dpi=180)
plt.close(fig)
print("Saved biot_number.png")

# --- Plot 2: T_lumped vs T_FD_avg ---
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(t_min, avg_T_FD,  color="tab:blue",   linewidth=1.2, label="FD Model (avg)")
ax.plot(t_min, T_lumped,  color="tab:orange",  linewidth=1.2,
        linestyle="--", label="Lumped Capacitance")
ax.set_xlabel("Time [min]")
ax.set_ylabel("Temperature [K]")
ax.set_title("Lumped Capacitance vs FD Average Temperature")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "lumped_vs_FD.png"), dpi=180)
plt.close(fig)
print("Saved lumped_vs_FD.png")

# --- Plot 3: Q_abs vs Q_emit ---
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(t_min, Q_abs     * 1e3, color="tab:red",   linewidth=1.0, label="Q_abs (all sources)")
ax.plot(t_min, Q_emit_FD * 1e3, color="tab:blue",  linewidth=1.0, label="Q_emit (FD avg T)")
ax.set_xlabel("Time [min]")
ax.set_ylabel("Heat Transfer [mW]")
ax.set_title("Absorbed vs Emitted Heat Transfer Over Time")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "Q_abs_vs_emit.png"), dpi=180)
plt.close(fig)
print("Saved Q_abs_vs_emit.png")

# --- Plot 4: Temperature error between lumped and FD ---
T_error = T_lumped - avg_T_FD
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(t_min, T_error, color="tab:green", linewidth=1.0)
ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
ax.set_xlabel("Time [min]")
ax.set_ylabel("T_lumped - T_FD [K]")
ax.set_title("Temperature Difference: Lumped vs FD Average")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "lumped_error.png"), dpi=180)
plt.close(fig)
print("Saved lumped_error.png")

# --- Plot 5: Effective radiation h over time ---
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(t_min, h_rad, color="tab:red", linewidth=1.2)
ax.set_xlabel("Time [min]")
ax.set_ylabel("h_rad [W/m²·K]")
ax.set_title("Effective Radiation Heat Transfer Coefficient vs Time")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "h_rad.png"), dpi=180)
plt.close(fig)
print("Saved h_rad.png")

# --- Plot 6: Net Q over time ---
Q_net = Q_abs - Q_emit_FD
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(t_min, Q_net * 1e3, color="tab:gray", linewidth=1.0)
ax.fill_between(t_min, 0, Q_net * 1e3,
                where=Q_net > 0, alpha=0.4, color="tab:red",  label="Net gain")
ax.fill_between(t_min, 0, Q_net * 1e3,
                where=Q_net < 0, alpha=0.4, color="tab:blue", label="Net loss")
ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
ax.set_xlabel("Time [min]")
ax.set_ylabel("Net Heat Transfer [mW]")
ax.set_title("Net Heat Transfer vs Time  (Q_abs − Q_emit)")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "Q_net.png"), dpi=180)
plt.close(fig)
print("Saved Q_net.png")

print("\nDone. All 6 plots saved to", OUT_DIR)
print(f"  Final T_lumped  = {T_lumped[-1]:.2f} K")
print(f"  Final T_FD_avg  = {avg_T_FD[-1]:.2f} K")
print(f"  Max |error|     = {np.abs(T_error).max():.3f} K")


