import numpy as np
import scipy.sparse as sparse
from OrbitHTModelFunctions import (
    q_cond, q_emmit, is_internal, tle_to_gcrs_state, inverse_index,
    precompute_rad_table, interp_rad_table
)
import time 
from astropy.time import Time
from astropy import units as u
from scipy.io import savemat
import os
import h5py

#Simulation Dimenstions
#Mesh Size
x_span = 1e-1 #m
y_span = 1e-1 #m
z_span = 1e-1 #m

V = x_span*y_span*z_span

#Wall Thickness
i_th = 10e-3 #m
j_th = 10e-3 #m
k_th = 10e-3 #m

V_int = (x_span-2*i_th)*(y_span-2*j_th)*(z_span-2*k_th)

#Cell Spacing
dx = 5e-3 #m
dy = 5e-3 #m
dz = 5e-3 #m

dV = dx*dy*dz

nx = x_span/dx 
ny = y_span/dy 
nz = z_span/dz 

n_count = nx*nz*ny

#if n_count != int(n_count):
#    print(f"nx:{nx} ,ny:{ny} ,nz:{nz}")
#    raise ValueError("Domain Not evenly Split. Make Sure n_span/dn are Integers.")

nx = round(nx)
ny = round(ny)
nz = round(nz)
n_count = round(n_count)
count_vols = round(n_count-(V_int/dV))
print("Full Cell Dims:",nx,ny,nz)
print("Full Cell Count:",n_count)
print("Empty Cell Count:",round(V_int/dV))
print("Occupied Cell Count:",count_vols)
print("Percent Full:",100*round(n_count-(V_int/dV))/n_count,"%")


#Assume Aluminum 6061-T6
k_alum = 167 #w/mk
rho_alum = 2700 #kg/m3
eps_alum = 0.05
cp_alum = 896 #j/kg/k
alpha_alum = 0.05
iso_space_heat_gen = 0 #W
print("Occupied Mass:", count_vols*dV*rho_alum, "[kg]")

T_init = 265 #k

T_prev = np.full(count_vols, T_init, dtype=np.float64)
T_next = T_prev.copy()

A_dir = np.array([dy*dz, dy*dz, dx*dz, dx*dz, dx*dy, dx*dy])
dl_dir = np.array([dx, dx, dy, dy, dz, dz])

t_orbit = 95*60
N_orbits = 30

t_max = 3600*24*30 #seconds
dt = 0.5
iter_max = int(t_max/dt) 
r = dt / (rho_alum*dx*dy*dz*cp_alum)
print(f"r Parameter [TimeStep]:{r}")

t0 = Time("2026-02-22T12:00:00", scale="utc")

#TLE for Satellite Trajectory
line1 = "1 25544U 98067A   19343.69339541  .00001764  00000-0  38792-4 0  9991"
line2 = "2 25544  51.6439 211.2001 0007417  17.6667  85.6398 15.50103472202482"

#Index[+i,-i,+j,-j,+k,-k]
#BC Codes
#[1=Conduction,2=Internal Cell]
#[-1=+i_rad, -2=-i_rad] 
#[-3=+j_rad ,-4=-j_rad]
#[-5=+k_rad ,-6=-k_rad]
NB = np.full((n_count, 6), -1, dtype=np.int32)
BC = np.zeros((n_count, 6), dtype=np.int8)

st = time.time()
#Neighbor and Boundary Condition Matrix Setup
shell_to_full = []
full_to_shell = np.full(n_count, -1, dtype=np.int32)

shell_count = 0
for k in range(nz):
    for j in range(ny):
        for i in range(nx):

            if is_internal(i, j, k,i_th,j_th,k_th,dx,dy,dz,x_span,y_span,z_span):
                continue

            # cell ID for Mapping
            c_n = i + j*nx + k*nx*ny

            shell_to_full.append(c_n)
            full_to_shell[c_n] = shell_count
            shell_count += 1

        

            if i < nx-1:    #<+i,0,0>
                NB[c_n,0] = -1 if is_internal(i+1, j, k,i_th,j_th,k_th,dx,dy,dz,x_span,y_span,z_span) else c_n + 1
                BC[c_n,0] =  2 if is_internal(i+1, j, k,i_th,j_th,k_th,dx,dy,dz,x_span,y_span,z_span) else 1
            else:
                NB[c_n,0] = -1
                BC[c_n,0] = -1

            if i > 0:       #<-i,0,0>
                NB[c_n,1] = -1 if is_internal(i-1, j, k,i_th,j_th,k_th,dx,dy,dz,x_span,y_span,z_span) else c_n - 1
                BC[c_n,1] =  2 if is_internal(i-1, j, k,i_th,j_th,k_th,dx,dy,dz,x_span,y_span,z_span) else 1
            else:
                NB[c_n,1] = -1
                BC[c_n,1] = -2

            if j < ny-1:    #<+j,0,0>
                NB[c_n,2] = -1 if is_internal(i, j+1, k,i_th,j_th,k_th,dx,dy,dz,x_span,y_span,z_span) else c_n + nx
                BC[c_n,2] =  2 if is_internal(i, j+1, k,i_th,j_th,k_th,dx,dy,dz,x_span,y_span,z_span) else 1
            else:
                NB[c_n,2] = -1
                BC[c_n,2] = -3

            if j > 0:       #<-j,0,0>
                NB[c_n,3] = -1 if is_internal(i, j-1, k,i_th,j_th,k_th,dx,dy,dz,x_span,y_span,z_span) else c_n - nx
                BC[c_n,3] =  2 if is_internal(i, j-1, k,i_th,j_th,k_th,dx,dy,dz,x_span,y_span,z_span) else 1
            else:
                NB[c_n,3] = -1
                BC[c_n,3] = -4

            if k < nz-1:    #<+k,0,0>
                NB[c_n,4] = -1 if is_internal(i, j, k+1,i_th,j_th,k_th,dx,dy,dz,x_span,y_span,z_span) else c_n + nx*ny
                BC[c_n,4] =  2 if is_internal(i, j, k+1,i_th,j_th,k_th,dx,dy,dz,x_span,y_span,z_span) else 1
            else:
                NB[c_n,4] = -1
                BC[c_n,4] = -5

            if k > 0:       #<-k,0,0>
                NB[c_n,5] = -1 if is_internal(i, j, k-1,i_th,j_th,k_th,dx,dy,dz,x_span,y_span,z_span) else c_n - nx*ny
                BC[c_n,5] =  2 if is_internal(i, j, k-1,i_th,j_th,k_th,dx,dy,dz,x_span,y_span,z_span) else 1
            else:
                NB[c_n,5] = -1
                BC[c_n,5] = -6



#A = sparse.csr_array(shape=[count_vols,count_vols],dtype=np.float64)
NB_new = np.full((count_vols, 6), -1, dtype=np.int32)
BC_new = np.zeros((count_vols, 6), dtype=np.int8)

count = 0
for c in range(n_count):
        if full_to_shell[c] == -1:
            continue
        for dir in range(6):
            if NB[c][dir] >= 0:
                NB_new[count][dir] = full_to_shell[NB[c][dir]]
            BC_new[count][dir] = BC[c][dir]
        count += 1

et = time.time()
print(f"After {et-st}s Boundary and Neighbor Cells Created")

ext_faces_count = np.count_nonzero(np.array(BC) < 0)
print("number of Exterior Faces:", ext_faces_count)
print("Exterior Area from Sim [m2]:",dx*dy*ext_faces_count)
int_faces_count = np.count_nonzero(np.array(BC)==2)



t_end = iter_max * dt


#Setup Over Time For Simulation

#Data Saving Block

output_dir = "sim_outputs"
os.makedirs(output_dir, exist_ok=True)

h5_path = os.path.join(output_dir, "thermal_results.h5")

if os.path.exists(h5_path):
    os.remove(h5_path)

save_every = 10000
n_saves_max = (iter_max + save_every - 1) // save_every

h5f = h5py.File(h5_path, "a")

T_dset = h5f.create_dataset(
    "T_store",
    shape=(0, count_vols),
    maxshape=(None, count_vols),
    dtype=np.float64,
    chunks=(1, count_vols)
)

time_dset = h5f.create_dataset(
    "time_sec",
    shape=(0,),
    maxshape=(None,),
    dtype=np.float64,
    chunks=(1,)
)

step_dset = h5f.create_dataset(
    "step",
    shape=(0,),
    maxshape=(None,),
    dtype=np.int64,
    chunks=(1,)
)

avgT_dset = h5f.create_dataset(
    "avg_T",
    shape=(0,),
    maxshape=(None,),
    dtype=np.float64,
    chunks=(1,)
)

h5f.attrs["dt"] = dt
h5f.attrs["nx"] = nx
h5f.attrs["ny"] = ny
h5f.attrs["nz"] = nz
h5f.attrs["count_vols"] = count_vols
h5f.attrs["save_every"] = save_every
h5f.create_dataset("shell_to_full", data=shell_to_full)

save_idx = 0


rad_sample_times, rad_table = precompute_rad_table(
    line1, line2, t0, t_end, 20, k
)
et = time.time()
print(f"After {et-st}s Rad Created")

st = time.time()

sigma     = 5.670e-8
cond_coef = k_alum * A_dir / dl_dir

# Pre-compute static masks — BC_new never changes, no need to rebuild each step
cond_masks     = []
cond_nb        = []
rad_masks      = []
rad_lookup_idx = []
any_cond       = []
any_rad        = []

for _d in range(6):
    _bc = BC_new[:, _d]
    _mc = (_bc == 1)
    cond_masks.append(_mc)
    cond_nb.append(NB_new[_mc, _d].copy())
    any_cond.append(bool(_mc.any()))
    _mr = (_bc < 0)
    rad_masks.append(_mr)
    rad_lookup_idx.append((-_bc[_mr] - 1).copy())
    any_rad.append(bool(_mr.any()))

# Pre-allocate work arrays — zero in-place each step instead of re-allocating
q_net        = np.zeros(count_vols)
q_rad_lookup = np.zeros(6)

# Radiation only needs re-interpolating at the rad-table's own sample cadence (20 s)
rad_update_every = max(1, int(60.0 / dt)) #Units [s]

for t in range(iter_max):
    tsec = t * dt

    if t % rad_update_every == 0:
        for fi in range(6):
            q_rad_lookup[fi] = (alpha_alum * A_dir[fi]
                                * np.interp(tsec, rad_sample_times, rad_table[:, fi]))

    emit_base_all = eps_alum * sigma * T_prev**4

    q_net[:] = 0.0

    for d in range(6):
        #Loop Through Directions and Sum Up heat transfer Q_net = 
        if any_cond[d]:
            q_net[cond_masks[d]] += -cond_coef[d] * (T_prev[cond_nb[d]] - T_prev[cond_masks[d]])
        if any_rad[d]:
            q_net[rad_masks[d]] += (emit_base_all[rad_masks[d]] * A_dir[d]
                                    - q_rad_lookup[rad_lookup_idx[d]])

    T_next[:] = T_prev - r * q_net

    if t % save_every == 0:
        avg_T_now = float(np.mean(T_prev))

        T_dset.resize((save_idx + 1, count_vols))
        T_dset[save_idx, :] = T_prev

        time_dset.resize((save_idx + 1,))
        time_dset[save_idx] = tsec

        step_dset.resize((save_idx + 1,))
        step_dset[save_idx] = t

        avgT_dset.resize((save_idx + 1,))
        avgT_dset[save_idx] = avg_T_now

        h5f.flush()
        et = time.time()
        print(f"[SAVE] step {t}  t={tsec:.1f}s | {tsec/3600:.2f}Hrs T̄={avg_T_now:.2f}K  "
              f"real_time={et-st:.1f}s  TTR={tsec/(et-st+1e-9):.2f}x")
        save_idx += 1

    T_prev, T_next = T_next, T_prev

h5f.close()
print(f"Saved results to {h5_path}")

#print (NB)
#print("_____")
#print(BC)





