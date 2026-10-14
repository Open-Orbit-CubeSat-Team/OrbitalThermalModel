import numpy as np
from sgp4.api import Satrec
from astropy import units as u
from astropy.coordinates import (
    TEME,
    GCRS,
    CartesianRepresentation,
    CartesianDifferential,
    get_sun,
)


def is_internal(i, j, k, i_th, j_th, k_th, dx, dy, dz, x_span, y_span, z_span):
    coord = [(i + 0.5) * dx, (j + 0.5) * dy, (k + 0.5) * dz]
    return (
        i_th < coord[0] < x_span - i_th
        and j_th < coord[1] < y_span - j_th
        and k_th < coord[2] < z_span - k_th
    )


def q_cond(k, A, Tl, T, dl):
    return -k * A * (Tl - T) / dl


def q_emmit(epsilon, A, T):
    steph = 5.670e-8
    return epsilon * A * steph * T**4


def compute_F_sE(delta, k):
    k = np.clip(k, -1.0, 1.0)
    delta = np.clip(delta, 0.0, np.pi)

    ck = np.arccos(k)

    if 0.0 <= delta <= ck:
        F = k**2 * np.cos(delta)

    elif ck < delta < np.pi - ck:
        Fa = k**2 * np.cos(delta)
        Fb = np.pi/2 - ((1 - k**2) * (k**2 - np.cos(delta)**2))**0.5
        Fc = -np.arcsin((1 - k**2)**0.5 / np.sin(delta))
        Fd = -k**2 * np.cos(delta) * np.arccos((1/k**2 - 1)**0.5 * (1/np.tan(delta)))
        F = Fa + (1/np.pi) * (Fb + Fc + Fd)

    elif np.pi - ck <= delta <= np.pi:
        F = 0.0

    else:
        print("Delta Error =", delta)
        print("k Error =", k)
        print("ck =", ck)
        raise ValueError("delta must be in [0, pi]")

    return F


def unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("Zero-length vector cannot be normalized.")
    return v / n


def compute_radiation_terms(r_sat, r_sun, n_surface, k, r_earth=6378.137):
    r_sat = np.asarray(r_sat, dtype=float)
    r_sun = np.asarray(r_sun, dtype=float)
    n_surface = np.asarray(n_surface, dtype=float)

    d = r_sun - r_sat
    dd = np.dot(d, d)

    if dd == 0:
        raise ValueError("Satellite and Sun vectors produce zero LOS vector.")

    lam = -np.dot(r_sat, d) / dd
    lam = np.clip(lam, 0.0, 1.0)

    r_close = r_sat + lam * d
    dmin = np.linalg.norm(r_close)
    sunlit = dmin >= r_earth

    s_hat = unit(r_sun - r_sat)
    e_hat = -unit(r_sat)
    n_hat = unit(n_surface)

    cos_theta = max(0.0, np.dot(n_hat, s_hat)) if sunlit else 0.0

    cos_delta = np.clip(np.dot(n_hat, e_hat), -1.0, 1.0)
    delta = np.arccos(cos_delta)
    F_sE = compute_F_sE(delta, k)

    cos_phi = np.dot(e_hat, s_hat)
    F_sa = F_sE * max(0.0, cos_phi) if sunlit else 0.0

    return sunlit, cos_theta, F_sE, F_sa


def q_abs(alpha, A, r_sat, r_sun, n_surface, k,
          sun_flux=1366.0, earth_ir=237.0, r_earth=6378.137, albedo=0.3):
    _, F_SS, F_SE, F_SA = compute_radiation_terms(
        r_sat, r_sun, n_surface, k, r_earth=r_earth
    )

    q_sun = sun_flux * A * alpha * F_SS
    q_earth = earth_ir * A * alpha * F_SE
    q_albedo = sun_flux * A * alpha * albedo * F_SA

    return q_sun + q_earth + q_albedo


def tle_to_gcrs_state(line1, line2, t):
    sat = Satrec.twoline2rv(line1, line2)

    jd, fr = t.jd1, t.jd2
    err, r_km, v_km_s = sat.sgp4(jd, fr)
    if err != 0:
        raise ValueError(f"SGP4 propagation failed with code {err}")

    teme = TEME(
        CartesianRepresentation(
            r_km * u.km,
            differentials=CartesianDifferential(v_km_s * u.km / u.s),
        ),
        obstime=t,
    )

    gcrs = teme.transform_to(GCRS(obstime=t))
    r_sat = gcrs.cartesian.xyz.to(u.km).value
    v_sat = gcrs.velocity.d_xyz.to(u.km / u.s).value

    sun = get_sun(t)
    r_sun = sun.cartesian.xyz.to(u.km).value

    return r_sat, v_sat, r_sun


def bc_code_to_normal(bc_code, r_sat, v_sat):
    e_hat = -unit(r_sat)
    v_hat = unit(v_sat)
    c_hat = unit(np.cross(e_hat, v_hat))

    if bc_code == -1:
        return e_hat
    elif bc_code == -2:
        return -e_hat
    elif bc_code == -3:
        return v_hat
    elif bc_code == -4:
        return -v_hat
    elif bc_code == -5:
        return c_hat
    elif bc_code == -6:
        return -c_hat
    else:
        raise ValueError(f"Invalid radiative BC code: {bc_code}")


def q_abs_bc(alpha, A, bc_code, t, line1, line2, k,
             sun_flux=1366.0, earth_ir=237.0, r_earth=6378.137):
    r_sat, v_sat, r_sun = tle_to_gcrs_state(line1, line2, t)
    n_surface = bc_code_to_normal(bc_code, r_sat, v_sat)

    return q_abs(
        alpha, A, r_sat, r_sun, n_surface, k,
        sun_flux=sun_flux, earth_ir=earth_ir, r_earth=r_earth
    )

def q_abs_geom(alpha, A, bc_code, r_sat, v_sat, r_sun, k,
               sun_flux=1366.0, earth_ir=237.0, r_earth=6378.137):
    n_surface = bc_code_to_normal(bc_code, r_sat, v_sat)
    return q_abs(
        alpha, A, r_sat, r_sun, n_surface, k,
        sun_flux=sun_flux, earth_ir=earth_ir, r_earth=r_earth
    )

def inverse_index(c, nx, ny):
    i = c % nx
    j = (c // nx) % ny
    k = c // (nx * ny)
    return i, j, k

def q_abs_geom_unit(bc_code, r_sat, v_sat, r_sun, k,
                    sun_flux=1366.0, earth_ir=237.0, r_earth=6378.137):
    n_surface = bc_code_to_normal(bc_code, r_sat, v_sat)
    return q_abs(
        1.0, 1.0, r_sat, r_sun, n_surface, k,
        sun_flux=sun_flux, earth_ir=earth_ir, r_earth=r_earth
    )

def precompute_rad_table(line1, line2, t0, t_end, sample_dt, k,
                         sun_flux=1366.0, earth_ir=237.0, r_earth=6378.137):
    sample_times = np.arange(0.0, t_end + sample_dt, sample_dt, dtype=np.float64)
    n_samp = len(sample_times)
    rad_table = np.zeros((n_samp, 6), dtype=np.float64)

    # Build the full time array once
    t_arr = t0 + sample_times * u.s

    # Parse TLE once and batch-propagate all sample times in one SGP4 call
    sat = Satrec.twoline2rv(line1, line2)
    errs, r_km, v_km_s = sat.sgp4_array(t_arr.jd1, t_arr.jd2)
    # r_km, v_km_s: (n_samp, 3)

    # One batch TEME → GCRS transform (single astropy call instead of n_samp calls)
    teme = TEME(
        CartesianRepresentation(
            r_km.T * u.km,
            differentials=CartesianDifferential(v_km_s.T * u.km / u.s),
        ),
        obstime=t_arr,
    )
    gcrs = teme.transform_to(GCRS(obstime=t_arr))
    r_sat_all = gcrs.cartesian.xyz.to(u.km).value.T          # (n_samp, 3)
    v_sat_all = gcrs.velocity.d_xyz.to(u.km / u.s).value.T   # (n_samp, 3)

    # One batch sun-position call
    r_sun_all = get_sun(t_arr).cartesian.xyz.to(u.km).value.T  # (n_samp, 3)

    # Fill radiation table — orbital geometry is now free; only the flux math loops
    for n in range(n_samp):
        if errs[n] != 0:
            continue
        for m, bc in enumerate([-1, -2, -3, -4, -5, -6]):
            rad_table[n, m] = q_abs_geom_unit(
                bc, r_sat_all[n], v_sat_all[n], r_sun_all[n], k,
                sun_flux=sun_flux, earth_ir=earth_ir, r_earth=r_earth
            )

    return sample_times, rad_table

def interp_rad_table(sample_times, rad_table, tsec, bc_code, alpha, A):
    col = -bc_code - 1   # -1->0, -2->1, ... -6->5
    q_unit = np.interp(tsec, sample_times, rad_table[:, col])
    return alpha * A * q_unit