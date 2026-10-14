import numpy as np
from sgp4.api import Satrec
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import TEME, GCRS, CartesianRepresentation, CartesianDifferential, get_sun
import matplotlib.pyplot as plt
from matplotlib import cm

from functions import compute_radiation_terms

r_earth = 6.371e3
alt_orbit = 6e2
k = r_earth / (r_earth + alt_orbit)

sun_rad = 1366
earth_rad = 237
albedo_factor = 0.3

boltz = 5.67e-8
eps_al = 0.05
alpha_al = 0.05

eps_sl = 0.8
panel_efficiency = 0.3
alpha_sl = 0.9 * (1 - panel_efficiency)

eps_sat = [eps_al, eps_al, eps_sl, eps_sl, eps_sl, eps_sl]
alpha_sat = [alpha_al, alpha_al, alpha_sl, alpha_sl, alpha_sl, alpha_sl]

height = 0.20
width = 0.10
thick = 0.10
A_sat = [width * thick, width * thick, width * height, width * height, thick * height, thick * height]


line1 = "1 25544U 98067A   19343.69339541  .00001764  00000-0  38792-4 0  9991"
line2 = "2 25544  51.6439 211.2001 0007417  17.6667  85.6398 15.50103472202482"

sat = Satrec.twoline2rv(line1, line2)

t0 = Time("2026-02-22T12:00:00", scale="utc")
minutes = np.linspace(0, 95, 200)

xs, ys, zs = [], [], []
U, V, W = [], [], []

sun_dirs = []
body_frames = []

FSE_hist = []
FSS_hist = []
FSA_hist = []
Q_dot_hist = []
valid_minutes = []

for dt_min in minutes:
    t = t0 + dt_min * u.min
    jd, fr = t.jd1, t.jd2

    sun = get_sun(t)
    sun_gcrs = np.array([
        np.float64(sun.cartesian.xyz.to(u.km)[0]),
        np.float64(sun.cartesian.xyz.to(u.km)[1]),
        np.float64(sun.cartesian.xyz.to(u.km)[2])
    ])

    err, r_km, v_km_s = sat.sgp4(jd, fr)
    if err != 0:
        continue

    teme = TEME(
        CartesianRepresentation(
            r_km * u.km,
            differentials=CartesianDifferential(v_km_s * u.km / u.s)
        ),
        obstime=t
    )

    gcrs = teme.transform_to(GCRS(obstime=t))
    r = gcrs.cartesian.xyz.to(u.km).value
    v = gcrs.velocity.d_xyz.to(u.km / u.s).value

    sE_n = -r / np.linalg.norm(r)
    v_norm = v / np.linalg.norm(v)

    cross_n = np.cross(sE_n, v_norm)
    cross_n /= np.linalg.norm(cross_n)

    sat_vec_norms = np.zeros((6, 3))
    sat_vec_norms[0] = sE_n
    sat_vec_norms[1] = -sE_n
    sat_vec_norms[2] = v_norm
    sat_vec_norms[3] = -v_norm
    sat_vec_norms[4] = cross_n
    sat_vec_norms[5] = -cross_n

    xs.append(r[0])
    ys.append(r[1])
    zs.append(r[2])
    U.append(v[0])
    V.append(v[1])
    W.append(v[2])

    sun_dirs.append(sun_gcrs / np.linalg.norm(sun_gcrs))
    body_frames.append(sat_vec_norms.copy())

    FSS_now = np.zeros(6)
    FSE_now = np.zeros(6)
    FSA_now = np.zeros(6)

    q_sun = 0
    q_ert = 0
    q_alb = 0
    q_emit = 0

    for n in range(6):
        loc_n = sat_vec_norms[n]
        lit, F_SS, F_SE, F_SA = compute_radiation_terms(r, sun_gcrs, loc_n, k)

        q_sun += sun_rad * A_sat[n] * alpha_sat[n] * F_SS
        q_ert += earth_rad * A_sat[n] * alpha_sat[n] * F_SE
        q_alb += sun_rad * A_sat[n] * alpha_sat[n] * albedo_factor * F_SA

        FSS_now[n] = F_SS
        FSE_now[n] = F_SE
        FSA_now[n] = F_SA

    FSS_hist.append(FSS_now)
    FSE_hist.append(FSE_now)
    FSA_hist.append(FSA_now)
    Q_dot_hist.append(q_sun + q_ert + q_alb - q_emit)
    valid_minutes.append(dt_min)

xs = np.array(xs)
ys = np.array(ys)
zs = np.array(zs)
U = np.array(U)
V = np.array(V)
W = np.array(W)

FSS_hist = np.array(FSS_hist)
FSE_hist = np.array(FSE_hist)
FSA_hist = np.array(FSA_hist)
Q_dot_hist = np.array(Q_dot_hist)
valid_minutes = np.array(valid_minutes)
sun_dirs = np.array(sun_dirs)
body_frames = np.array(body_frames)

t_sec = valid_minutes * 60.0
Q_net_cum = np.zeros_like(Q_dot_hist)
Q_net_cum[1:] = np.cumsum(0.5 * (Q_dot_hist[1:] + Q_dot_hist[:-1]) * np.diff(t_sec))
Q_net_total = Q_net_cum[-1]

print("Total integrated net heat [J] =", Q_net_total)
print("Average Heat Dissipation Req for Equilibrium [W] =", -Q_net_total / t_sec[-1])

# choose snapshot index here
idx = -1

sat_gcrs = np.array([xs[idx], ys[idx], zs[idx]])
sat_vel = np.array([U[idx], V[idx], W[idx]])
sun_dir = sun_dirs[idx]
sat_vec_norms = body_frames[idx]

u_sphere = np.linspace(0, 2 * np.pi, 200)
v_sphere = np.linspace(0, np.pi, 100)

Xe = r_earth * np.outer(np.cos(u_sphere), np.sin(v_sphere))
Ye = r_earth * np.outer(np.sin(u_sphere), np.sin(v_sphere))
Ze = r_earth * np.outer(np.ones_like(u_sphere), np.cos(v_sphere))

Nx = Xe / r_earth
Ny = Ye / r_earth
Nz = Ze / r_earth

intensity = Nx * sun_dir[0] + Ny * sun_dir[1] + Nz * sun_dir[2]
intensity = np.clip(intensity, 0, 1)

earth_base = cm.get_cmap("gist_earth")(0.35 + 0.45 * intensity)
night_mask = (Nx * sun_dir[0] + Ny * sun_dir[1] + Nz * sun_dir[2]) < 0
earth_base[..., :3] *= (0.08 + 0.92 * intensity[..., None])
earth_base[night_mask, :3] *= 0.15

cloud = 0.08 * np.sin(6 * np.arctan2(Ye, Xe)) * np.sin(5 * np.arccos(Ze / r_earth))
cloud = np.clip(cloud, 0, 0.08)
earth_base[..., :3] = np.clip(earth_base[..., :3] + cloud[..., None] * intensity[..., None], 0, 1)

fig = plt.figure(figsize=(10, 10), facecolor="black")
ax = fig.add_subplot(111, projection="3d")
ax.set_facecolor("black")

ax.plot_surface(
    Xe, Ye, Ze,
    facecolors=earth_base,
    rstride=1, cstride=1,
    linewidth=0,
    antialiased=False,
    shade=False
)

ax.scatter(
    sat_gcrs[0], sat_gcrs[1], sat_gcrs[2],
    s=80, c="red", edgecolors="white", label="Satellite"
)

elev = 20
azim = 35
ax.view_init(elev=elev, azim=azim)

az = np.deg2rad(azim)
el = np.deg2rad(elev)
view_dir = np.array([
    np.cos(el) * np.cos(az),
    np.cos(el) * np.sin(az),
    np.sin(el)
])

orbit_pts = np.column_stack((xs, ys, zs))
front_mask = orbit_pts @ view_dir > 0

ax.plot(xs, ys, zs, color="cyan", linewidth=1.0, alpha=0.20)

start = None
for i, is_front in enumerate(front_mask):
    if is_front and start is None:
        start = i
    elif not is_front and start is not None:
        ax.plot(xs[start:i], ys[start:i], zs[start:i], color="cyan", linewidth=2.2)
        start = None
if start is not None:
    ax.plot(xs[start:], ys[start:], zs[start:], color="cyan", linewidth=2.2)

ax.plot([], [], [], color="cyan", linewidth=2.2, label="Orbit")

sun_arrow_len = 700
ax.quiver(
    sat_gcrs[0], sat_gcrs[1], sat_gcrs[2],
    sun_dir[0] * sun_arrow_len,
    sun_dir[1] * sun_arrow_len,
    sun_dir[2] * sun_arrow_len,
    color="yellow", linewidth=2, label="Sun Direction"
)

vel_arrow_len = 700
vel_unit = sat_vel / np.linalg.norm(sat_vel)
ax.quiver(
    sat_gcrs[0], sat_gcrs[1], sat_gcrs[2],
    vel_unit[0] * vel_arrow_len,
    vel_unit[1] * vel_arrow_len,
    vel_unit[2] * vel_arrow_len,
    color="red", linewidth=2.5, label="Velocity"
)

frame_len = 500
for i in range(6):
    ax.quiver(
        sat_gcrs[0], sat_gcrs[1], sat_gcrs[2],
        sat_vec_norms[i, 0] * frame_len,
        sat_vec_norms[i, 1] * frame_len,
        sat_vec_norms[i, 2] * frame_len,
        color="lime", alpha=0.85
    )

ax.plot(
    [0, sat_gcrs[0]],
    [0, sat_gcrs[1]],
    [0, sat_gcrs[2]],
    color="white", alpha=0.25, linestyle="--"
)

r_plot = r_earth + alt_orbit + 1500
ax.set_xlim(-r_plot, r_plot)
ax.set_ylim(-r_plot, r_plot)
ax.set_zlim(-r_plot, r_plot)
ax.set_box_aspect([1, 1, 1])

ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

try:
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("black")
    ax.yaxis.pane.set_edgecolor("black")
    ax.zaxis.pane.set_edgecolor("black")
except Exception:
    pass

ax.set_title("Satellite Orbit Around Earth", color="white", pad=20)
leg = ax.legend()
for txt in leg.get_texts():
    txt.set_color("white")

plt.tight_layout()
plt.show()