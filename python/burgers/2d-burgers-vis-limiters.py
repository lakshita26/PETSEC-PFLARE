import sys
import warnings
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
import numpy as np

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# -----------------------------
# 1. Parameters & Grid Setup
# -----------------------------
Nx, Ny = 64, 64     # 2D Grid size (64x64)
L = 1.0             
dx = L / Nx
dy = L / Ny
dt = 0.002          # Smaller timestep for 2D stability
T_final = 0.5       

# --- CONTROL PANEL ---
# Set nu = 0.01 for Viscous
# Set nu = 0.0  for Inviscid
nu = 0.01           
# ---------------------

# Create 2D DMDA
da = PETSc.DMDA().create(dim=2, sizes=[Nx, Ny], 
                         boundary_type=('periodic', 'periodic'), 
                         stencil_width=2)
da.setUp()

def vanleer(r):
    """Van Leer Limiter Function"""
    return (r + abs(r)) / (1.0 + abs(r))

# -----------------------------
# 2. FormFunction (2D Physics)
# -----------------------------
def FormFunction(snes, U, F_vec):
    U_loc = da.createLocalVec()
    da.globalToLocal(U, U_loc)
    
    u = da.getVecArray(U_loc)
    f = da.getVecArray(F_vec)
    u_old = da.getVecArray(prev_u)
    
    (xs, xe), (ys, ye) = da.getRanges()
    
    # Loop over 2D Grid
    for j in range(ys, ye):
        for i in range(xs, xe):
            
            # === X-DIRECTION FLUX (MUSCL) ===
            denom_i = (u[j, i+1] - u[j, i])
            r_i = (u[j, i] - u[j, i-1]) / denom_i if abs(denom_i) > 1e-12 else 0.0
            u_right = u[j, i] + 0.5 * vanleer(r_i) * (u[j, i+1] - u[j, i])

            denom_im1 = (u[j, i] - u[j, i-1])
            r_im1 = (u[j, i-1] - u[j, i-2]) / denom_im1 if abs(denom_im1) > 1e-12 else 0.0
            u_left = u[j, i-1] + 0.5 * vanleer(r_im1) * (u[j, i] - u[j, i-1])

            Flux_x_right = 0.5 * u_right**2
            Flux_x_left  = 0.5 * u_left**2
            dFx_dx = (Flux_x_right - Flux_x_left) / dx

            # === Y-DIRECTION FLUX (MUSCL) ===
            denom_j = (u[j+1, i] - u[j, i])
            r_j = (u[j, i] - u[j-1, i]) / denom_j if abs(denom_j) > 1e-12 else 0.0
            u_top = u[j, i] + 0.5 * vanleer(r_j) * (u[j+1, i] - u[j, i])

            denom_jm1 = (u[j, i] - u[j-1, i])
            r_jm1 = (u[j-1, i] - u[j-2, i]) / denom_jm1 if abs(denom_jm1) > 1e-12 else 0.0
            u_bot = u[j-1, i] + 0.5 * vanleer(r_jm1) * (u[j, i] - u[j-1, i])

            Flux_y_top = 0.5 * u_top**2
            Flux_y_bot = 0.5 * u_bot**2
            dFy_dy = (Flux_y_top - Flux_y_bot) / dy

            # === DIFFUSION (Central Difference) ===
            diff_x = nu * (u[j, i+1] - 2*u[j, i] + u[j, i-1]) / (dx**2)
            diff_y = nu * (u[j+1, i] - 2*u[j, i] + u[j-1, i]) / (dy**2)

            # === TOTAL RESIDUAL ===
            f[j, i] = (u[j, i] - u_old[j, i]) + dt * (dFx_dx + dFy_dy - diff_x - diff_y)

# -----------------------------
# 3. FormJacobian (Fixed: Standard Global Indices)
# -----------------------------
def FormJacobian(snes, U, J, P):
    """
    Jacobian using Global Indices to avoid 'AttributeError'.
    """
    U_loc = da.createLocalVec()
    da.globalToLocal(U, U_loc)
    u = da.getVecArray(U_loc)
    
    P.zeroEntries()
    (xs, xe), (ys, ye) = da.getRanges()
    
    # Precompute diffusion factors
    Dx = dt * nu / (dx**2)
    Dy = dt * nu / (dy**2)

    for j in range(ys, ye):
        for i in range(xs, xe):
            # 1. Global Row ID
            row_global = j * Nx + i
            
            # 2. Physics
            a_local = u[j, i]
            adv_diag_x = (dt / dx) * a_local
            adv_off_L  = -(dt / dx) * u[j, i-1]
            adv_diag_y = (dt / dy) * a_local
            adv_off_B  = -(dt / dy) * u[j-1, i]

            # 3. Matrix Values
            val_diag = 1.0 + adv_diag_x + adv_diag_y + 2.0*Dx + 2.0*Dy
            val_L = adv_off_L - Dx
            val_R = - Dx
            val_B = adv_off_B - Dy
            val_T = - Dy

            # 4. Periodic Neighbor Logic
            i_L = (i - 1) % Nx; j_L = j
            i_R = (i + 1) % Nx; j_R = j
            i_B = i;            j_B = (j - 1) % Ny
            i_T = i;            j_T = (j + 1) % Ny

            # 5. Neighbor Global IDs
            col_L = j_L * Nx + i_L
            col_R = j_R * Nx + i_R
            col_B = j_B * Nx + i_B
            col_T = j_T * Nx + i_T
            
            # 6. Set Values
            cols = [col_L, col_R, col_B, col_T, row_global]
            vals = [val_L, val_R, val_B, val_T, val_diag]
            
            P.setValues([row_global], cols, vals, PETSc.InsertMode.INSERT_VALUES)
            
    P.assemble()
    if J != P: J.assemble()
    return True

# -----------------------------
# 4. Initialization
# -----------------------------
u = da.createGlobalVec()
prev_u = u.duplicate()
F = u.duplicate()
P_mat = da.createMatrix()

# Initial Condition: 2D Gaussian Bump
u_arr = da.getVecArray(u)
(xs, xe), (ys, ye) = da.getRanges()

for j in range(ys, ye):
    for i in range(xs, xe):
        x = i * dx
        y = j * dy
        r2 = (x - 0.5)**2 + (y - 0.5)**2
        u_arr[j, i] = 0.5 + 0.5 * np.exp(-r2 / 0.05)

u.copy(prev_u)

# -----------------------------
# 5. Solver Setup
# -----------------------------
snes = PETSc.SNES().create()
snes.setDM(da)
snes.setFunction(FormFunction, F)
snes.setJacobian(FormJacobian, P_mat)
snes.setFromOptions()

# -----------------------------
# 6. Time Loop
# -----------------------------
t = 0.0
total_snes = 0
total_ksp = 0

if PETSc.COMM_WORLD.getRank() == 0:
    print(f"2D Burgers (Viscosity nu={nu})")
    print("-" * 55)
    print(f"{'Time':>10} | {'SNES Its':>10} | {'KSP Its':>10}")
    print("-" * 55)

while t < T_final:
    snes.solve(None, u)
    
    s_its = snes.getIterationNumber()
    k_its = snes.getLinearSolveIterations()
    total_snes += s_its
    total_ksp += k_its
    
    if PETSc.COMM_WORLD.getRank() == 0:
        print(f"{t:10.3f} | {s_its:10} | {k_its:10}")
        
    u.copy(prev_u)
    t += dt

if PETSc.COMM_WORLD.getRank() == 0:
    print("-" * 55)
    print(f"Simulation Complete.")
    print(f"Total SNES: {total_snes} | Total KSP: {total_ksp}")