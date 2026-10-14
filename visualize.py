"""
visualize.py  —  Combined orbital + thermal visualization for the CubeSat model.

Outputs written to sim_outputs/:
  avg_temperature.png        average shell temperature vs time
  temperature_envelope.png   min / mean / max temperature band vs time
  heat_flux_faces.png        absorbed heat flux per face vs time (6 panels)
  thermal_orbit.gif          side-by-side animated orbit globe + cubesat thermal
"""

import os
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import imageio.v2 as imageio
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import (
    TEME, GCRS, CartesianRepresentation, CartesianDifferential, get_sun,
)
from sgp4.api import Satrec
from PIL import Image

from OrbitHTModelFunctions import precompute_rad_table

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
H5_PATH    = "sim_outputs/thermal_results.h5"
OUT_DIR    = "sim_outputs"

FRAME_SKIP = 10    # every Nth saved snapshot → one animation frame
FPS        = 10    # animation playback rate

CS_ELEV, CS_AZIM = 22, 38   # cubesat scatter view
ORB_ELEV         = 20        # globe view elevation

TLE_LINE1 = "1 25544U 98067A   19343.69339541  .00001764  00000-0  38792-4 0  9991"
TLE_LINE2 = "2 25544  51.6439 211.2001 0007417  17.6667  85.6398 15.50103472202482"
T0        = Time("2026-02-22T12:00:00", scale="utc")

R_EARTH   = 6371.0   # km
ALT_ORBIT = 600.0    # km

# Must match CubeSatFiniteDiff.py
DX = DY = DZ  = 2e-3   # m
ALPHA_ALUM    = 0.05
EPS_ALUM      = 0.05
SIGMA         = 5.670e-8
A_DIR = np.array([DY*DZ, DY*DZ, DX*DZ, DX*DZ, DX*DY, DX*DY])

FACE_LABELS = ["+i  nadir", "-i  zenith", "+j  velocity",
               "-j  anti-vel", "+k  cross-track", "-k  anti-cross"]
FACE_COLORS = ["tab:red","tab:blue","tab:orange","tab:cyan","tab:green","tab:purple"]

# ── LOAD METADATA (no large arrays yet) ───────────────────────────────────────
if not os.path.exists(H5_PATH):
    raise FileNotFoundError(
        f"HDF5 file not found at '{H5_PATH}'. Run CubeSatFiniteDiff.py first."

        
    )

_TEXTURE_PATH = "image.png"
try:
    _earth_img = np.array(Image.open(_TEXTURE_PATH).convert("RGB"), dtype=np.float32) / 255.0
    _TEX_H, _TEX_W = _earth_img.shape[:2]
    print(f"Loaded Earth texture: {_TEX_W}x{_TEX_H}")
except FileNotFoundError:
    raise FileNotFoundError(
        f"Blue Marble texture not found at '{_TEXTURE_PATH}'.\n"
        "Download from: https://eoimages.gsfc.nasa.gov/images/imagerecords/57000/57752/land_shallow_topo_2048.jpg\n"
        "and save as textures/blue_marble.jpg"
    )


with h5py.File(H5_PATH, "r") as f:
    n_total       = f["T_store"].shape[0]
    n_shell       = f["T_store"].shape[1]
    time_sec_all  = f["time_sec"][:]
    avg_T_all     = f["avg_T"][:]
    shell_to_full = f["shell_to_full"][:]
    nx = int(f.attrs["nx"])
    ny = int(f.attrs["ny"])
    nz = int(f.attrs["nz"])

if n_total == 0:
    raise ValueError(
        f"HDF5 file at '{H5_PATH}' contains 0 saved snapshots.\n"
        "Run CubeSatFiniteDiff.py first to generate simulation data."
    )

print(f"Snapshots : {n_total}   shell cells : {n_shell}")
print(f"Grid      : {nx} × {ny} × {nz}")
print(f"Time span : {time_sec_all[0]:.0f} – {time_sec_all[-1]:.0f} s  "
      f"({time_sec_all[-1]/60:.1f} min)")

# Voxel grid indices for 3-D scatter
full_idx = shell_to_full.astype(np.int64)
ix = full_idx % nx
iy = (full_idx // nx) % ny
iz = full_idx // (nx * ny)

# ── RADIATION HISTORY (vectorised interpolation) ──────────────────────────────
print("Precomputing radiation table …")
k_orbit = R_EARTH / (R_EARTH + ALT_ORBIT)
t_end_sim = float(time_sec_all[-1])

rad_sample_times, rad_table = precompute_rad_table(
    TLE_LINE1, TLE_LINE2, T0, t_end_sim, 20.0, k_orbit
)

# q_abs_hist[i, face] = absorbed W for one cell face at snapshot i
q_abs_hist = np.zeros((n_total, 6), dtype=np.float64)
for fi in range(6):
    q_unit = np.interp(time_sec_all.astype(float), rad_sample_times, rad_table[:, fi])
    q_abs_hist[:, fi] = ALPHA_ALUM * A_DIR[fi] * q_unit

# Net absorbed vs emitted (using avg temperature as proxy for emission estimate)
q_net_hist = q_abs_hist.sum(axis=1)   # W per cell, total absorbed across all faces

# ── TEMPERATURE ENVELOPE (chunked read — memory-safe) ─────────────────────────
print("Computing temperature envelope …")
T_min_hist = np.zeros(n_total, dtype=np.float32)
T_max_hist = np.zeros(n_total, dtype=np.float32)
CHUNK = 200
with h5py.File(H5_PATH, "r") as f:
    for start in range(0, n_total, CHUNK):
        end   = min(start + CHUNK, n_total)
        block = f["T_store"][start:end]
        T_min_hist[start:end] = block.min(axis=1)
        T_max_hist[start:end] = block.max(axis=1)

# ── STATIC PLOT 1 — Average temperature ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(time_sec_all / 60, avg_T_all, linewidth=1.5, color="tab:red")
ax.set_xlabel("Time [min]")
ax.set_ylabel("Temperature [K]")
ax.set_title("CubeSat Average Temperature vs Time")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "avg_temperature.png"), dpi=180)
plt.close(fig)
print("Saved avg_temperature.png")

# ── STATIC PLOT 2 — Temperature envelope ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
ax.fill_between(time_sec_all / 60, T_min_hist, T_max_hist,
                alpha=0.25, color="tab:blue", label="Min – Max range")
ax.plot(time_sec_all / 60, avg_T_all, color="tab:blue", linewidth=1.5, label="Mean")
ax.set_xlabel("Time [min]")
ax.set_ylabel("Temperature [K]")
ax.set_title("CubeSat Temperature Envelope")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "temperature_envelope.png"), dpi=180)
plt.close(fig)
print("Saved temperature_envelope.png")

# ── STATIC PLOT 3 — Heat flux per face + net ──────────────────────────────────
fig = plt.figure(figsize=(13, 9))
gs  = fig.add_gridspec(4, 2, hspace=0.45, wspace=0.35)

for fi in range(6):
    row, col = divmod(fi, 2)
    ax = fig.add_subplot(gs[row, col])
    ax.plot(time_sec_all / 60, q_abs_hist[:, fi] * 1e3,
            color=FACE_COLORS[fi], linewidth=0.8)
    ax.set_title(FACE_LABELS[fi], fontsize=9)
    ax.set_ylabel("Absorbed [mW]", fontsize=8)
    ax.grid(True, alpha=0.3)
    if row == 2:
        ax.set_xlabel("Time [min]", fontsize=8)

ax_net = fig.add_subplot(gs[3, :])
ax_net.plot(time_sec_all / 60, q_net_hist * 1e3,
            color="tab:gray", linewidth=1.0)
ax_net.fill_between(time_sec_all / 60, 0, q_net_hist * 1e3,
                    where=q_net_hist > 0, alpha=0.4, color="tab:red",  label="Net gain")
ax_net.fill_between(time_sec_all / 60, 0, q_net_hist * 1e3,
                    where=q_net_hist < 0, alpha=0.4, color="tab:blue", label="Net loss")
ax_net.axhline(0, color="gray", linewidth=0.6, linestyle="--")
ax_net.set_xlabel("Time [min]", fontsize=9)
ax_net.set_ylabel("Net absorbed [mW]", fontsize=8)
ax_net.set_title("Total absorbed across all faces", fontsize=9)
ax_net.legend(fontsize=8)
ax_net.grid(True, alpha=0.3)

fig.suptitle("Absorbed Heat Flux per Face", fontsize=12)
fig.savefig(os.path.join(OUT_DIR, "heat_flux_faces.png"), dpi=180)
plt.close(fig)
print("Saved heat_flux_faces.png")

# ── STATIC PLOT 4 — Q_in vs Q_out ────────────────────────────────────────────
A_outer = A_DIR.sum()   # total outer surface area [m²]
q_out_hist = EPS_ALUM * SIGMA * A_outer * avg_T_all**4   # W per shell cell (avg T proxy)
q_in_hist  = q_net_hist                                   # total absorbed [W]

q_net_thermal = q_in_hist - q_out_hist   # positive = heating, negative = cooling

fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

ax_top.plot(time_sec_all / 60, q_in_hist  * 1e3, color="tab:orange", linewidth=1.2, label="$Q_{in}$ (absorbed)")
ax_top.plot(time_sec_all / 60, q_out_hist * 1e3, color="tab:blue",   linewidth=1.2, label="$Q_{out}$ (emitted)")
ax_top.set_ylabel("Power [mW]")
ax_top.set_title("CubeSat $Q_{in}$ vs $Q_{out}$ vs Time")
ax_top.legend()
ax_top.grid(True, alpha=0.3)

t_min = time_sec_all / 60
ax_bot.plot(t_min, q_net_thermal * 1e3, color="tab:gray", linewidth=1.0)
ax_bot.fill_between(t_min, 0, q_net_thermal * 1e3,
                    where=q_net_thermal >= 0, alpha=0.4, color="tab:red",  label="Net gain")
ax_bot.fill_between(t_min, 0, q_net_thermal * 1e3,
                    where=q_net_thermal <  0, alpha=0.4, color="tab:blue", label="Net loss")
ax_bot.axhline(0, color="gray", linewidth=0.6, linestyle="--")
ax_bot.set_xlabel("Time [min]")
ax_bot.set_ylabel("$Q_{net}$ [mW]")
ax_bot.set_title("Net Heat ($Q_{in} - Q_{out}$)")
ax_bot.legend()
ax_bot.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "q_in_out.png"), dpi=180)
plt.close(fig)
print("Saved q_in_out.png")

# ── SELECT ANIMATION FRAMES ────────────────────────────────────────────────────
anim_idx   = np.arange(0, n_total, FRAME_SKIP)
n_anim     = len(anim_idx)
time_anim  = time_sec_all[anim_idx]
avg_T_anim = avg_T_all[anim_idx]

print(f"Loading {n_anim} animation frames (skip={FRAME_SKIP}) …")
T_frames = np.zeros((n_anim, n_shell), dtype=np.float32)
with h5py.File(H5_PATH, "r") as f:
    for out_i, fi in enumerate(anim_idx):
        T_frames[out_i] = f["T_store"][fi]

T_vmin = float(np.percentile(T_frames, 2))
T_vmax = float(np.percentile(T_frames, 98))
norm_T = Normalize(vmin=T_vmin, vmax=T_vmax)
cmap_T = cm.inferno

# ── ORBITAL POSITIONS at animation times ──────────────────────────────────────
print("Computing orbital positions for animation …")
sgp4_sat = Satrec.twoline2rv(TLE_LINE1, TLE_LINE2)

r_anim   = np.zeros((n_anim, 3))
sun_anim = np.zeros((n_anim, 3))

for i, tsec in enumerate(time_anim):
    t = T0 + float(tsec) * u.s
    err, r_km, v_km_s = sgp4_sat.sgp4(t.jd1, t.jd2)
    if err != 0:
        r_anim[i] = r_anim[max(i - 1, 0)]
        continue
    teme = TEME(
        CartesianRepresentation(
            r_km * u.km,
            differentials=CartesianDifferential(v_km_s * u.km / u.s),
        ),
        obstime=t,
    )
    gcrs = teme.transform_to(GCRS(obstime=t))
    r_anim[i] = gcrs.cartesian.xyz.to(u.km).value
    sv = get_sun(t).cartesian.xyz.to(u.km).value
    sun_anim[i] = sv / np.linalg.norm(sv)

# Dense ghost trail (every ~30 s) for background
n_trail    = max(int(t_end_sim / 30) + 1, 2)
trail_tsec = np.linspace(0.0, t_end_sim, n_trail)
r_trail    = np.zeros((n_trail, 3))
for i, tsec in enumerate(trail_tsec):
    t = T0 + float(tsec) * u.s
    err, r_km, v_km_s = sgp4_sat.sgp4(t.jd1, t.jd2)
    if err != 0:
        r_trail[i] = r_trail[max(i - 1, 0)]
        continue
    teme = TEME(
        CartesianRepresentation(
            r_km * u.km,
            differentials=CartesianDifferential(v_km_s * u.km / u.s),
        ),
        obstime=t,
    )
    gcrs = teme.transform_to(GCRS(obstime=t))
    r_trail[i] = gcrs.cartesian.xyz.to(u.km).value

# ── ORBIT OCCLUSION HELPERS ───────────────────────────────────────────────────
def _cam_dir(elev_deg, azim_deg):
    """Unit vector from scene origin toward the camera."""
    az = np.deg2rad(azim_deg)
    el = np.deg2rad(elev_deg)
    return np.array([np.cos(el) * np.cos(az),
                     np.cos(el) * np.sin(az),
                     np.sin(el)])

def _visible_mask(points, cam_dir, r_earth):
    """True for points NOT occluded by a sphere of radius r_earth at origin."""
    d      = points @ cam_dir                       # projection along view axis
    perp2  = np.maximum(0.0, (points**2).sum(1) - d**2)  # perpendicular dist²
    return ~((d < 0) & (perp2 < r_earth**2))

def _plot_visible(ax, points, visible, **kwargs):
    """Draw only the contiguous visible segments of a polyline.
    Accepts both 2-D (N×2) and 3-D (N×3) point arrays."""
    n, i = len(points), 0
    while i < n:
        if visible[i]:
            j = i + 1
            while j < n and visible[j]:
                j += 1
            if j - i > 1:
                if points.shape[1] == 2:
                    ax.plot(points[i:j, 0], points[i:j, 1], **kwargs)
                else:
                    ax.plot(points[i:j, 0], points[i:j, 1], points[i:j, 2], **kwargs)
            i = j
        else:
            i += 1


def _ortho_basis(cam_dir):
    """Return (right, up) orthonormal camera-plane basis vectors."""
    z_world = np.array([0.0, 0.0, 1.0])
    hint = np.array([0.0, 1.0, 0.0]) if abs(np.dot(cam_dir, z_world)) > 0.999 else z_world
    right = np.cross(hint, cam_dir)
    right /= np.linalg.norm(right)
    up = np.cross(cam_dir, right)
    up /= np.linalg.norm(up)
    return right, up


def _project_2d(points, right, up):
    """Orthographic projection of (N,3) world points → (N,2) screen coords."""
    return np.column_stack([points @ right, points @ up])

def _render_earth_disk(right, up, forward, sun_d, R, res=300):
    """
    Ray-cast an RGBA orthographic image of Earth using the Blue Marble texture.

    For each pixel inside the Earth disk:
      1. Compute the 3D surface point on the sphere
      2. Convert to lat/lon
      3. Sample the equirectangular texture
      4. Apply Lambertian day/night shading

    Returns an (res, res, 4) float32 RGBA array.
    extent=[-R, R, -R, R] for ax.imshow.
    Alpha=0 outside the disk.
    """
    lin    = np.linspace(-R, R, res)
    ug, vg = np.meshgrid(lin, lin)
    r2     = ug**2 + vg**2
    inside = r2 <= R * R

    # 3D surface point for each pixel inside the disk
    depth         = np.zeros_like(ug)
    depth[inside] = np.sqrt(R * R - r2[inside])

    # World-space 3D coords on the sphere surface
    # P = u*right + v*up + depth*forward  →  shape (res, res, 3)
    P = (ug[..., None] * right +
         vg[..., None] * up   +
         depth[..., None] * forward)

    # Unit normal on sphere
    Pn = np.zeros_like(P)
    Pn[inside] = P[inside] / R

    # ── Lat / Lon from surface normal ─────────────────────────────────────────
    # Pn is in GCRS-like frame: x toward vernal equinox, z toward north pole
    # lat = arcsin(z),  lon = arctan2(y, x)
    lat = np.arcsin(np.clip(Pn[..., 2], -1.0, 1.0))          # [-pi/2, pi/2]
    lon = np.arctan2(Pn[..., 1], Pn[..., 0])                  # [-pi,   pi  ]

    # Map to texture pixel coordinates
    # lon: -pi → 0,  pi → W-1
    # lat:  pi/2 → 0,  -pi/2 → H-1  (image top = north)
    tx = ((lon + np.pi) / (2 * np.pi) * (_TEX_W - 1)).astype(np.int32)
    ty = ((np.pi / 2 - lat) / np.pi   * (_TEX_H - 1)).astype(np.int32)

    tx = np.clip(tx, 0, _TEX_W - 1)
    ty = np.clip(ty, 0, _TEX_H - 1)

    # ── Sample texture ────────────────────────────────────────────────────────
    rgba = np.zeros((res, res, 4), dtype=np.float32)
    rgba[inside, :3] = _earth_img[ty[inside], tx[inside]]

    # ── Lambertian day/night shading ──────────────────────────────────────────
    intensity        = np.zeros((res, res), dtype=np.float32)
    intensity[inside] = np.clip(Pn[inside] @ sun_d, 0.0, 1.0)

    # Night side darkening: ambient 0.08, full lit = 1.0
    dark          = np.zeros((res, res), dtype=np.float32)
    dark[inside]  = 0.08 + 0.92 * intensity[inside]
    rgba[..., :3] *= dark[..., None]

    # Soft terminator blend — slightly soften the day/night boundary
    terminator        = np.zeros((res, res), dtype=np.float32)
    terminator[inside] = np.clip(
        (Pn[inside] @ sun_d + 0.05) / 0.15, 0.0, 1.0
    )
    rgba[inside, :3] *= (0.3 + 0.7 * terminator[inside, None])

    # Alpha mask
    rgba[..., 3] = inside.astype(np.float32)

    return rgba


# ── RENDER ANIMATION ──────────────────────────────────────────────────────────
print(f"Rendering {n_anim} frames …")
frames_out = []
BG = "#0d0d0d"

for fi in range(n_anim):
    temp   = T_frames[fi]
    tsec   = float(time_anim[fi])
    r_sat  = r_anim[fi]
    sun_d  = sun_anim[fi]
    avg_t  = float(avg_T_anim[fi])

    trail_end = max(1, int(tsec / t_end_sim * n_trail))

    azim_now = 35 + 0.12 * fi
    cam      = _cam_dir(ORB_ELEV, azim_now)

    vis_full      = _visible_mask(r_trail,             cam, R_EARTH)
    vis_completed = _visible_mask(r_trail[:trail_end], cam, R_EARTH)

    fig = plt.figure(figsize=(14, 6.5), facecolor=BG)

    # ── LEFT panel: orbit globe (2-D orthographic projection) ────────────────
    right2d, up2d = _ortho_basis(cam)
    forward2d     = cam

    earth_rgba = _render_earth_disk(right2d, up2d, forward2d, sun_d, R_EARTH, res=300)

    # Project all 3-D positions to 2-D screen coords
    trail_2d_full      = _project_2d(r_trail,             right2d, up2d)
    trail_2d_completed = _project_2d(r_trail[:trail_end], right2d, up2d)
    sat_2d             = _project_2d(r_sat[None, :],      right2d, up2d)[0]
    sun_tip_2d         = _project_2d((r_sat + sun_d * 650)[None, :], right2d, up2d)[0]

    ax_orb = fig.add_subplot(121)
    ax_orb.set_facecolor(BG)
    ax_orb.set_aspect("equal")
    r_lim_2d = R_EARTH + ALT_ORBIT + 1300
    ax_orb.set_xlim(-r_lim_2d, r_lim_2d)
    ax_orb.set_ylim(-r_lim_2d, r_lim_2d)
    ax_orb.axis("off")

    # Draw order — painter's algorithm (back → front)

    # 1. Occluded back-side ghost trail (very faint, drawn before Earth disk)
    _plot_visible(ax_orb, trail_2d_full, ~vis_full,
                  color="cyan", linewidth=0.4, alpha=0.07, zorder=1)

    # 2. Earth disk — alpha mask keeps corners transparent
    ax_orb.imshow(earth_rgba,
                  extent=[-R_EARTH, R_EARTH, -R_EARTH, R_EARTH],
                  origin="lower", aspect="equal",
                  interpolation="bilinear", zorder=2)

    # 3. Front-side ghost full orbit
    _plot_visible(ax_orb, trail_2d_full, vis_full,
                  color="cyan", linewidth=0.5, alpha=0.18, zorder=3)

    # 4. Front-side completed trail (bright)
    _plot_visible(ax_orb, trail_2d_completed, vis_completed,
                  color="cyan", linewidth=1.6, alpha=0.9, zorder=4)

    # 5. Satellite marker
    sat_c = cmap_T(norm_T(avg_t))
    ax_orb.scatter(sat_2d[0], sat_2d[1],
                   s=70, c=[sat_c], edgecolors="white", linewidths=0.5, zorder=5)

    # 6. Sun direction arrow
    ax_orb.annotate("",
                    xy=(sun_tip_2d[0], sun_tip_2d[1]),
                    xytext=(sat_2d[0], sat_2d[1]),
                    arrowprops=dict(arrowstyle="-|>", color="yellow",
                                   lw=1.2, mutation_scale=12),
                    zorder=6)

    ax_orb.set_title(f"t = {tsec/60:.1f} min", color="white", pad=4, fontsize=10)

    # ── RIGHT panel: CubeSat thermal scatter ──────────────────────────────────
    ax_cs = fig.add_subplot(122, projection="3d")
    ax_cs.set_facecolor(BG)

    sc = ax_cs.scatter(ix, iy, iz, c=temp, s=14,
                       cmap="inferno", vmin=T_vmin, vmax=T_vmax,
                       depthshade=True)

    ax_cs.set_xlabel("i", color="white", labelpad=1, fontsize=7)
    ax_cs.set_ylabel("j", color="white", labelpad=1, fontsize=7)
    ax_cs.set_zlabel("k", color="white", labelpad=1, fontsize=7)
    ax_cs.tick_params(colors="white", labelsize=6)
    ax_cs.set_box_aspect([nx, ny, nz])
    ax_cs.view_init(elev=CS_ELEV, azim=CS_AZIM)
    try:
        for pane in [ax_cs.xaxis.pane, ax_cs.yaxis.pane, ax_cs.zaxis.pane]:
            pane.fill = False
            pane.set_edgecolor("#1c1c1c")
    except Exception:
        pass
    ax_cs.set_title(
        f"T̄ = {avg_t:.1f} K    min {temp.min():.1f}   max {temp.max():.1f}",
        color="white", pad=4, fontsize=9,
    )

    cbar = fig.colorbar(sc, ax=ax_cs, pad=0.08, shrink=0.72)
    cbar.set_label("Temperature [K]", color="white", fontsize=8)
    cbar.ax.yaxis.set_tick_params(color="white", labelsize=7)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    fig.tight_layout(pad=0.8)
    fig.canvas.draw()
    w_px, h_px = fig.canvas.get_width_height()
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h_px, w_px, 4)
    frames_out.append(img[:, :, :3].copy())
    plt.close(fig)

    if fi % 20 == 0:
        print(f"  {fi:4d}/{n_anim}  ({100*fi/n_anim:.0f}%)  "
              f"t={tsec/60:.1f} min  T̄={avg_t:.1f} K")

gif_path = os.path.join(OUT_DIR, "thermal_orbit.gif")
imageio.mimsave(gif_path, frames_out, duration=1.0 / FPS, loop=0)

print(f"\nDone.")
print(f"  {gif_path}")
print(f"  {OUT_DIR}/avg_temperature.png")
print(f"  {OUT_DIR}/temperature_envelope.png")
print(f"  {OUT_DIR}/heat_flux_faces.png")
print(f"  {OUT_DIR}/q_in_out.png")
