import os
import h5py
import numpy as np
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -----------------------------
# CONFIG
# -----------------------------
h5_path = "sim_outputs/thermal_results.h5"
avg_plot_path = "avg_temperature_.png"
gif_path = "temperature_3d.gif"

frame_skip = 10          # increase to skip more saved frames
fps = 6
use_deltaT = True       # True = animate T - T_initial, usually much more visible
point_size = 18
elev = 22
azim = 38

# -----------------------------
# LOAD
# -----------------------------
if not os.path.exists(h5_path):
    raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

with h5py.File(h5_path, "r") as f:
    print("Top-level keys:")
    for k in f.keys():
        print(" ", k)

    if "T_store" not in f:
        raise KeyError("Dataset 'T_store' not found.")

    T_store = f["T_store"][:]

    if "time_sec" in f:
        time_sec = f["time_sec"][:]
    else:
        time_sec = np.arange(T_store.shape[0], dtype=float)

    if "avg_T" in f:
        avg_T = f["avg_T"][:]
    else:
        avg_T = np.mean(T_store, axis=1)

    nx = int(f.attrs["nx"]) if "nx" in f.attrs else None
    ny = int(f.attrs["ny"]) if "ny" in f.attrs else None
    nz = int(f.attrs["nz"]) if "nz" in f.attrs else None

    shell_to_full = f["shell_to_full"][:] if "shell_to_full" in f else None

if nx is None or ny is None or nz is None:
    raise ValueError("Missing attrs 'nx', 'ny', 'nz' in HDF5 file.")

if shell_to_full is None:
    raise ValueError("Missing dataset 'shell_to_full'. Save it during the simulation output step.")

if T_store.ndim == 1:
    T_store = T_store[np.newaxis, :]
elif T_store.ndim != 2:
    raise ValueError(f"T_store must be 1D or 2D, got shape {T_store.shape}")

n_frames, n_shell = T_store.shape
if len(shell_to_full) != n_shell:
    raise ValueError(
        f"Mismatch: len(shell_to_full)={len(shell_to_full)} but T_store.shape[1]={n_shell}"
    )

print(f"T_store shape: {T_store.shape}")
print(f"Grid dims: nx={nx}, ny={ny}, nz={nz}")

# -----------------------------
# SAVE AVG T PLOT
# -----------------------------
plt.figure(figsize=(8, 5))
plt.plot(time_sec, avg_T, linewidth=2)
plt.xlabel("Time [s]")
plt.ylabel("Average Temperature [K]")
plt.title("Average Temperature vs Time")
plt.grid(True)
plt.tight_layout()
plt.savefig(avg_plot_path, dpi=180)
plt.close()
print(f"Saved average temperature plot to: {avg_plot_path}")

# -----------------------------
# PRECOMPUTE 3D COORDINATES
# c = i + j*nx + k*nx*ny
# -----------------------------
full_idx = shell_to_full.astype(np.int64)

ix = full_idx % nx
iy = (full_idx // nx) % ny
iz = full_idx // (nx * ny)

# -----------------------------
# CHOOSE WHAT TO ANIMATE
# -----------------------------
if use_deltaT:
    field = T_store - T_store[0]
    cbar_label = "ΔT [K]"
    plot_title = "3D Temperature Change"
else:
    field = T_store
    cbar_label = "Temperature [K]"
    plot_title = "3D Temperature"

# robust scaling
vals = field[np.isfinite(field)]
if vals.size == 0:
    raise ValueError("No finite values found in temperature field.")

vmin = np.percentile(vals, 2)
vmax = np.percentile(vals, 98)

if abs(vmax - vmin) < 1e-12:
    vmin = np.min(vals)
    vmax = np.max(vals) + 1e-12

print(f"Color scale: {vmin:.6e} to {vmax:.6e}")

# -----------------------------
# BUILD FRAMES
# -----------------------------
frames = []

for n in range(0, n_frames, frame_skip):
    temp = field[n]

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(
        ix, iy, iz,
        c=temp,
        s=point_size,
        cmap="inferno",
        vmin=vmin,
        vmax=vmax
    )

    ax.set_xlabel("i")
    ax.set_ylabel("j")
    ax.set_zlabel("k")
    ax.set_title(f"{plot_title}\nframe={n}, t={time_sec[min(n, len(time_sec)-1)]:.3f} s")
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect([nx, ny, nz])

    cbar = fig.colorbar(sc, ax=ax, pad=0.08, shrink=0.8)
    cbar.set_label(cbar_label)

    fig.tight_layout()

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    frames.append(img[:, :, :3].copy())

    plt.close(fig)

if len(frames) == 0:
    raise ValueError("No frames created for GIF.")

duration = 1.0 / fps
imageio.mimsave(gif_path, frames, duration=duration, loop=0)
print(f"Saved 3D GIF to: {gif_path}")