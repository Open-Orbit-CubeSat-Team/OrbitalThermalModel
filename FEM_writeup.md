# From Finite Difference to Finite Element: CubeSat Thermal Model

---

## 1. What Problem Are We Solving?

Your CubeSat code solves the **transient heat equation** in 3D:

$$\rho c_p \frac{\partial T}{\partial t} = \nabla \cdot (k \nabla T) + \dot{q}$$

- Left side: thermal inertia — how fast the material stores/releases energy
- Right side first term: conduction through the material
- Right side second term: volumetric heat source (zero in your interior; radiation on the shell)

The boundary condition on outer faces is nonlinear radiation:

$$q_{BC} = \varepsilon \sigma T^4 - \alpha \, q_{abs}(t)$$

This is the governing equation for both FD and FEM. The difference is entirely in **how you discretize it**.

---

## 2. Finite Difference vs. Finite Element — The Core Distinction

### Finite Difference (what you have now)

FD approximates **derivatives directly** using Taylor series.

The spatial derivative is replaced by differences between neighboring node values:

$$\frac{\partial^2 T}{\partial x^2} \approx \frac{T_{i+1} - 2T_i + T_{i-1}}{\Delta x^2}$$

**Key characteristics:**
- Works on a **structured, uniform grid** — every cell is the same size and shape
- The unknowns live at **grid points** arranged in a regular lattice
- Easy to implement but geometry must map cleanly onto a rectangular mesh
- Boundary conditions are enforced by directly replacing equations at boundary nodes
- Your code does this explicitly: `BC[c_n, d] = -1` flags an outer face and you substitute the radiation flux directly into `q_net`

**What your explicit time loop is doing:**

$$T_i^{n+1} = T_i^n - \frac{\Delta t}{\rho c_p \Delta V} \sum_{\text{faces}} Q_{\text{face}}$$

Each cell's new temperature is a weighted sum of its neighbors — no linear system to solve. This is the **explicit (forward Euler) FD method**.

---

### Finite Element (what FEM does instead)

FEM does not approximate derivatives directly. Instead it asks:

> "What function $T(\mathbf{x})$ satisfies the PDE *on average* over each small region?"

**The two key ideas:**

**1. Weak form (variational form)**

Multiply the PDE by a test function $w(\mathbf{x})$ and integrate over the domain $\Omega$:

$$\int_\Omega w \, \rho c_p \frac{\partial T}{\partial t} \, dV = \int_\Omega w \, \nabla \cdot (k \nabla T) \, dV + \int_\Omega w \, \dot{q} \, dV$$

Apply integration by parts (Green's theorem) to the conduction term to move one derivative onto $w$:

$$\int_\Omega w \, \rho c_p \dot{T} \, dV + \int_\Omega k \nabla w \cdot \nabla T \, dV = \oint_{\partial \Omega} w \, k \nabla T \cdot \hat{n} \, dA + \int_\Omega w \, \dot{q} \, dV$$

The surface integral on the right is exactly where your **radiation boundary condition** enters naturally — you substitute $q_{BC} = \varepsilon \sigma T^4 - \alpha q_{abs}$ for the normal flux on outer faces.

This is called the **weak form**. It requires less smoothness from $T$ than the original PDE (only first derivatives instead of second), which is why FEM handles complex geometry and discontinuous material properties more gracefully.

**2. Shape functions (basis functions)**

The temperature field is approximated as a sum of simple polynomial functions:

$$T(\mathbf{x}, t) \approx \sum_{j=1}^{N} T_j(t) \, \phi_j(\mathbf{x})$$

where $\phi_j(\mathbf{x})$ is the **shape function** for node $j$ — a function that equals 1 at node $j$ and 0 at all other nodes. For a linear element this is just a tent function (piecewise linear).

Substituting into the weak form with $w = \phi_i$ produces a **system of ODEs**:

$$\mathbf{M} \dot{\mathbf{T}} + \mathbf{K} \mathbf{T} = \mathbf{f}(t)$$

| Matrix | Name | Role |
|--------|------|------|
| $\mathbf{M}$ | Mass matrix | Thermal inertia, $M_{ij} = \int_\Omega \rho c_p \phi_i \phi_j \, dV$ |
| $\mathbf{K}$ | Stiffness matrix | Conduction, $K_{ij} = \int_\Omega k \nabla\phi_i \cdot \nabla\phi_j \, dV$ |
| $\mathbf{f}$ | Load vector | Heat sources + boundary fluxes |

This is a **global linear system** (or nonlinear if BCs include $T^4$). You must solve it at every time step.

---

## 3. Side-by-Side Comparison

| | Finite Difference | Finite Element |
|---|---|---|
| Discretization | Derivatives → differences | PDE → integral (weak form) |
| Mesh | Structured, uniform grid | Unstructured, arbitrary shapes |
| Unknowns | Node temperatures on lattice | Nodal coefficients of shape functions |
| Result at each step | Explicit update formula | Linear system $\mathbf{M}\dot{T} + \mathbf{K}T = f$ |
| BCs | Substitute directly into node equation | Enter naturally through surface integral |
| Geometry | Rectangular domains only | Curved surfaces, holes, complex shapes |
| Complexity | Simpler to code | More infrastructure (assembly, solver) |

---

## 4. Translating Your CubeSat Code to FEM

### Step 1 — Mesh the geometry

Replace your `nx × ny × nz` voxel grid with a mesh of **hexahedral** (brick) or **tetrahedral** elements. For a thin-walled hollow cube you could use a single layer of 8-node hex elements through the wall thickness.

Each element has nodes at its corners. The mesh stores:
- `nodes[n, 3]` — x, y, z coordinate of each node
- `elements[e, 8]` — the 8 node IDs for each hex element (4 for a tet)

Your current shell detection (`is_internal`) goes away — you simply don't mesh the hollow interior.

### Step 2 — Compute element matrices

For each element $e$, compute the **element stiffness** and **element mass** matrices by numerical integration (Gauss quadrature):

```python
# For a single hex element with nodes at coords xe (8x3):
def element_matrices(xe, k_cond, rho, cp):
    Ke = np.zeros((8, 8))   # element stiffness
    Me = np.zeros((8, 8))   # element mass
    
    for gp, w in gauss_points_3d():          # 8 Gauss points for a hex
        N, dN_dxi = shape_functions(gp)      # shape funcs and derivatives
        J = dN_dxi @ xe                      # Jacobian (maps ref → physical)
        dN_dx = np.linalg.solve(J, dN_dxi)  # physical derivatives
        
        dV = abs(np.linalg.det(J)) * w
        Ke += k_cond * (dN_dx.T @ dN_dx) * dV
        Me += rho * cp * np.outer(N, N) * dV
    
    return Ke, Me
```

### Step 3 — Assemble the global system

Loop over all elements and scatter element matrices into global `K` and `M` (sparse matrices):

```python
K = sparse.lil_matrix((n_nodes, n_nodes))
M = sparse.lil_matrix((n_nodes, n_nodes))

for e, elem_nodes in enumerate(elements):
    Ke, Me = element_matrices(nodes[elem_nodes], k_alum, rho_alum, cp_alum)
    for i, ni in enumerate(elem_nodes):
        for j, nj in enumerate(elem_nodes):
            K[ni, nj] += Ke[i, j]
            M[ni, nj] += Me[i, j]
```

This replaces your `cond_coef` and neighbor-lookup loop entirely.

### Step 4 — Apply radiation boundary conditions

Identify the **outer surface faces** of your mesh (analogous to your `BC < 0` mask). For each outer face, compute a **surface flux vector** contribution to $\mathbf{f}$:

$$f_i^{rad} = \int_{\Gamma} \phi_i \left( \alpha \, q_{abs}(t) - \varepsilon \sigma T^4 \right) dA$$

The $T^4$ term makes this nonlinear. In practice you either:
- **Linearize**: evaluate $T^4$ using $T^n$ from the previous step (explicit treatment, same as your current code)
- **Iterate**: use Newton's method within each time step

```python
def radiation_flux_vector(face_nodes, T_face, q_absorbed, eps, sigma, alpha):
    f_rad = np.zeros(4)   # 4 nodes on a quad face
    for gp, w in gauss_points_2d():
        N, J2d = surface_shape_functions(face_nodes, gp)
        dA = abs(J2d) * w
        T_gp = N @ T_face
        q = alpha * q_absorbed - eps * sigma * T_gp**4
        f_rad += N * q * dA
    return f_rad
```

Your `q_rad_lookup` table works exactly the same way — you're just integrating the absorbed flux over each outer face with the shape functions as weights instead of applying it uniformly per cell.

### Step 5 — Time integration

The assembled system is:

$$\mathbf{M} \frac{\mathbf{T}^{n+1} - \mathbf{T}^n}{\Delta t} + \mathbf{K} \mathbf{T}^n = \mathbf{f}^n$$

Rearranged for explicit (forward Euler) — the same scheme your FD code uses:

$$\mathbf{T}^{n+1} = \mathbf{T}^n + \Delta t \, \mathbf{M}^{-1} \left( \mathbf{f}^n - \mathbf{K} \mathbf{T}^n \right)$$

With a **lumped mass matrix** (diagonal $M_{ii} = \sum_j M_{ij}$, zeroing off-diagonals), this inverts trivially — just one divide per node, exactly like your FD update:

```python
M_lumped = np.array(M.sum(axis=1)).flatten()   # row sums
T_next = T_prev + dt * (f - K @ T_prev) / M_lumped
```

This is the direct FEM equivalent of your current line:

```python
T_next[:] = T_prev - r * q_net   # your FD update
```

The `r` parameter in your code (`dt / (rho * dx*dy*dz * cp)`) is exactly `dt / M_lumped[i]` for a uniform mesh.

---

## 5. Why Bother with FEM for This Problem?

For a rectangular aluminum box, your FD code is entirely valid and more efficient. FEM becomes worthwhile when:

- The geometry becomes non-rectangular (curved panels, brackets, antenna mounts)
- Wall thickness varies across the structure
- You want to couple the thermal model to structural stress (thermoelastic analysis)
- You want to use commercial tools like Abaqus, ANSYS, or FEniCS, which all use FEM internally

For a CubeSat thermal model, the standard industry workflow is to mesh the structure in a CAD tool, export to an FEM thermal solver, and apply the orbital flux boundary conditions — which is exactly what your radiation model (`precompute_rad_table`, `bc_code_to_normal`) already computes correctly. The physics is the same; only the spatial discretization changes.
