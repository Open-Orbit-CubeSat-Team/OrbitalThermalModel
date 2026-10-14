import numpy as np
from sgp4.api import Satrec
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import TEME, GCRS, CartesianRepresentation, CartesianDifferential, get_sun
import matplotlib.pyplot as plt
from matplotlib import cm

from functions import compute_radiation_terms

r_earth = 6.371e3 #km
alt_orbit = 6e2 #km
k = r_earth / (r_earth + alt_orbit)

sun_rad = 1366 #w/m2
earth_rad = 237 #w/m2
albedo_factor = 0.3

boltz = 5.67e-8 #w/m2k4
eps_al = 0.05 #apsorptivity of Solar Panel
alpha_al = 0.05 #emmissivity of Al

eps_sl = 0.8
panel_efficiency = 0.2
alpha_sl = 0.9 * (1 - panel_efficiency)

eps_sat = [eps_al, eps_al, eps_sl, eps_sl, eps_sl, eps_sl]
alpha_sat = [alpha_al, alpha_al, alpha_sl, alpha_sl, alpha_sl, alpha_sl]

height = 1  #m2
width = 1  #m2
thick = 1 #m2
A_sat = [width * thick, width * thick, width * height, width * height, thick * height, thick * height]


line1 = "1 25544U 98067A   19343.69339541  .00001764  00000-0  38792-4 0  9991"
line2 = "2 25544  51.6439 211.2001 0007417  17.6667  85.6398 15.50103472202482"

line1 = "1 38771U 12049A   26160.29378457  .00000000  00000+0  36333-4 0 00017"
line2 = "2 38771  98.6504 211.9516 0002500  40.7295 258.0003 14.21430920712137"

sat = Satrec.twoline2rv(line1, line2)

t0 = Time("2026-02-22T12:00:00", scale="utc")
n_orbits = 2
minutes = np.linspace(0, 95*n_orbits, 200*n_orbits)

xs, ys, zs = [], [], []
U, V, W = [], [], []

sun_dirs = []
body_frames = []

FSE_hist = []
FSS_hist = []
FSA_hist = []
Q_dot_hist = []
Ei_hist = []
Ei_hist_t = []
valid_minutes = []

names = ["nadir","zenith","fwd","bwd","left","right"]

for dt_min in minutes:
    t = t0 + dt_min * u.min
    jd, fr = t.jd1, t.jd2

    sun = get_sun(t)
    sun_gcrs = np.array([
        np.float64(sun.cartesian.xyz.to(u.km)[0]),
        np.float64(sun.cartesian.xyz.to(u.km)[1]),
        np.float64(sun.cartesian.xyz.to(u.km)[2])
    ])
    sun_norm = sun_gcrs/np.linalg.norm(sun_gcrs)

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

    proj = sun_norm - (np.dot(sE_n, sun_norm) / np.dot(sE_n, sE_n)) * sE_n
    best_sun_dir = proj/np.linalg.norm(proj)

    #print(best_sun_dir)
    cross_n = np.cross(sE_n, best_sun_dir)
    cross_n /= np.linalg.norm(cross_n)

    sat_vec_norms = np.zeros((6, 3))
    sat_vec_norms[0] = sE_n
    sat_vec_norms[1] = -sE_n
    sat_vec_norms[2] = best_sun_dir
    sat_vec_norms[3] = -best_sun_dir
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

    Ei_side = np.zeros(6)

    for n in range(6):
        loc_n = sat_vec_norms[n]
        lit, F_SS, F_SE, F_SA = compute_radiation_terms(r, sun_gcrs, loc_n, k)
        
        # Incident Energy on each face, Normalized
        e_sun = sun_rad * F_SS
        e_ert = earth_rad * F_SE
        e_alb = sun_rad * albedo_factor * F_SA

        #Incident Radiation Absorbed by the Surface
        q_sun += e_sun * alpha_sat[n] *A_sat[n]  
        q_ert += e_ert * alpha_sat[n] *A_sat[n]  
        q_alb += e_alb * alpha_sat[n] * albedo_factor *A_sat[n]  

        Ei_side[n] += e_sun  + e_alb

        FSS_now[n] = F_SS
        FSE_now[n] = F_SE
        FSA_now[n] = F_SA

    Ei_hist_t.append(np.sum(Ei_side))

    Ei_hist.append(Ei_side)
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

Ei_hist_t = np.array(Ei_hist_t)
Ei_hist = np.array(Ei_hist)
FSS_hist = np.array(FSS_hist)
FSE_hist = np.array(FSE_hist)
FSA_hist = np.array(FSA_hist)
Q_dot_hist = np.array(Q_dot_hist)
valid_minutes = np.array(valid_minutes)
sun_dirs = np.array(sun_dirs)
body_frames = np.array(body_frames)

#Do some math on the arrays energy 
#Energy per U_SA = (0.1m)^2 = 1e-2 m^2
U_area = 1e-2
sp_hist = Ei_hist * panel_efficiency * U_area
sp_hist_t = Ei_hist_t * panel_efficiency * U_area

sp_avg_abs = sum(sp_hist_t)/len(sp_hist_t)

t_sec = valid_minutes * 60.0
Q_net_cum = np.zeros_like(Q_dot_hist)
Q_net_cum[1:] = np.cumsum(0.5 * (Q_dot_hist[1:] + Q_dot_hist[:-1]) * np.diff(t_sec))
Q_net_total = Q_net_cum[-1]

for n in range(6):
    print(f"Avg Energy Absorbed[w/U]:{np.sum(sp_hist[:,n])/len(sp_hist_t)} in the {names[n]} direction")

print("Average Absorbed Energy For Fully Covered [w/U]:",sp_avg_abs)
print("Total integrated net heat [J] =", Q_net_total)
print("Average Heat Dissipation Req for Equilibrium [W] =", -Q_net_total / t_sec[-1])

fig,ax = plt.subplots(nrows=3,ncols=2)

ax[0,0].plot(minutes,Ei_hist[:,0],color="red",label="nadir")
ax[0,0].plot(minutes,Ei_hist[:,1],color="orange",label="zenith")
ax[0,0].plot(minutes,Ei_hist[:,2],color="yellow",label="maroon")
ax[0,0].set_xlabel("W/m2")
ax[0,0].set_ylabel("Minutes")
ax[0,0].legend(loc='upper right') 

ax[1,0].plot(minutes,Ei_hist[:,3],color="green",label="bwd")
ax[1,0].plot(minutes,Ei_hist[:,4],color="blue",label="Left")
ax[1,0].plot(minutes,Ei_hist[:,5],color="purple",label="right")
ax[1,0].set_ylabel("W/m2")
ax[1,0].set_xlabel("Minutes")
ax[1,0].legend(loc='upper right') 

ax[2,0].plot(minutes,Ei_hist_t)
ax[2,0].set_ylabel("W/m2")
ax[2,0].set_xlabel("Minutes")

ax[0,1].plot(minutes,sp_hist[:,0],color="red")
ax[0,1].plot(minutes,sp_hist[:,1],color="orange")
ax[0,1].plot(minutes,sp_hist[:,2],color="maroon")
ax[0,1].set_xlabel("W/m2")
ax[0,1].set_ylabel("Minutes")
ax[0,1].legend(loc='upper right') 


ax[1,1].plot(minutes,sp_hist[:,3],color="green")
ax[1,1].plot(minutes,sp_hist[:,4],color="blue")
ax[1,1].plot(minutes,sp_hist[:,5],color="purple")
ax[1,1].set_ylabel("W/m2")
ax[1,1].set_xlabel("Minutes")


ax[2,1].plot(minutes,sp_hist_t)
ax[2,1].set_ylabel("W/m2")
ax[2,1].set_xlabel("Minutes")  

# 3. Add an overall canvas title and fix overlapping layout issues
fig.suptitle('incident Energy on Each Face (W/m2) for 1 orbit', fontsize=16)
plt.tight_layout()  # Automatically adjusts margins and spacing

# 4. Display the figure
plt.show()

