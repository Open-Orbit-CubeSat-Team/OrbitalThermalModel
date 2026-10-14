import numpy as np

def compute_F_sE(delta,k):
    ck = np.arccos(k)
    
    if (0<=delta<=ck):
        F=k**2*np.cos(delta)
    elif (ck<delta<np.pi-ck):
        Fa=k**2*np.cos(delta)
        Fb=(np.pi/2-((1-k**2)*(k**2-np.cos(delta)**2))**(0.5))
        Fc=-np.arcsin((1-k**2)**(0.5)/np.sin(delta))
        Fd= -k**2*np.cos(delta)*np.arccos((1/k**2-1)**(0.5)*(1/np.tan(delta)))
        
        F = Fa + (1/np.pi)*(Fb+Fc+Fd)
        
    elif (np.pi-ck <= delta <=np.pi):
        F=0
    else:
        print("Delta Error = ",delta)
        raise ValueError("delta must be in [0, pi]")
        
    return F

import numpy as np

def unit(v):
    return v / np.linalg.norm(v)

def compute_radiation_terms(r_sat, r_sun, n_surface, k, r_earth=6378.137):
    r_sat = np.asarray(r_sat, dtype=float)
    r_sun = np.asarray(r_sun, dtype=float)
    n_surface = np.asarray(n_surface, dtype=float)

    # Sunlight / eclipse test
    d = r_sun - r_sat
    dd = np.dot(d, d)

    if dd == 0:
        raise ValueError("Satellite and Sun vectors produce zero LOS vector.")

    lam = -np.dot(r_sat, d) / dd
    lam = np.clip(lam, 0.0, 1.0)

    r_close = r_sat + lam * d
    dmin = np.linalg.norm(r_close)
    sunlit = dmin >= r_earth

    # Unit vectors
    s_hat = unit(r_sun - r_sat)   # sat -> sun
    e_hat = -unit(r_sat)          # sat -> earth
    n_hat = unit(n_surface)       # surface normal

    # Direct solar incidence
    cos_theta = max(0.0, np.dot(n_hat, s_hat)) if sunlit else 0.0

    # Earth IR view factor
    cos_delta = np.clip(np.dot(n_hat, e_hat), -1.0, 1.0)
    delta = np.arccos(cos_delta)
    F_sE = compute_F_sE(delta, k)

    # Albedo term
    cos_phi = np.dot(e_hat, s_hat)
    F_sa = F_sE * max(0.0, cos_phi) if sunlit else 0.0

    return sunlit, cos_theta, F_sE, F_sa