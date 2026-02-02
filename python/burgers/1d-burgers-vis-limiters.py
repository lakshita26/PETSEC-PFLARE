import sys
import warnings
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
import numpy as np

# Suppress annoying NumPy/PETSc warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# -----------------------------
# 1. Parameters & Grid Setup
# -----------------------------
N = 256             
L = 1.0             
dx = L / N
dt = 0.005          
T_final = 0.5       

# Physics Parameter: Viscosity
nu = 0.01           

# DMDA Setup
da = PETSc.DMDA().create(dim=1, sizes=[N], boundary_type=('periodic',), stencil_width=2)
da.setUp()

def vanleer(r):
    """Van Leer Limiter Function"""
    return (r + abs(r)) / (1.0 + abs(r))

# -----------------------------
# 2. FormFunction (Physics Residual)
# -----------------------------
def FormFunction(snes, U, F_vec):
    U_loc = da.createLocalVec()
    da.globalToLocal(U, U_loc)
    
    u = da.getVecArray(U_loc)
    f = da.getVecArray(F_vec)
    u_old = da.getVecArray(prev_u)
    
    xs, xe = da.getRanges()[0]
    
    for i in range(xs, xe):
        # --- A. Advection Part (MUSCL) ---
        
        # Right Interface (i + 1/2)
        denom_i = (u[i+1] - u[i])
        r_i = (u[i] - u[i-1]) / denom_i if abs(denom_i) > 1e-12 else 0.0
        u_face_right = u[i] + 0.5 * vanleer(r_i) * (u[i+1] - u[i])

        # Left Interface (i - 1/2)
        denom_im1 = (u[i] - u[i-1])
        r_im1 = (u[i-1] - u[i-2]) / denom_im1 if abs(denom_im1) > 1e-12 else 0.0
        u_face_left = u[i-1] + 0.5 * vanleer(r_im1) * (u[i] - u[i-1])

        # Advection Fluxes (F = 0.5 * u^2)
        Flux_adv_right = 0.5 * u_face_right**2
        Flux_adv_left  = 0.5 * u_face_left**2
        
        # Advection Term: dF/dx
        term_advection = (Flux_adv_right - Flux_adv_left) / dx

        # --- B. Viscous Part (Central Difference) ---
        # d^2u/dx^2 approx (u[i+1] - 2u[i] + u[i-1]) / dx^2
        term_diffusion = nu * (u[i+1] - 2*u[i] + u[i-1]) / (dx**2)

        # --- C. Residual Calculation ---
        f[i] = (u[i] - u_old[i]) + dt * (term_advection - term_diffusion)

# -----------------------------
# 3. FormJacobian (Fixed: Handles Periodic Wrap)
# -----------------------------
def FormJacobian(snes, U, J, P):
    """
    Jacobian with Diffusion + Periodic Boundary Fix.
    """
    U_loc = da.createLocalVec()
    da.globalToLocal(U, U_loc)
    u = da.getVecArray(U_loc)
    
    P.zeroEntries()
    xs, xe = da.getRanges()[0]
    
    # Precompute constant diffusion factor
    diff_factor = dt * nu / (dx**2)

    for i in range(xs, xe):
        a_local = u[i]
        
        # --- Advection Contributions (Upwind) ---
        adv_diag = (dt / dx) * a_local
        adv_off_L = -(dt / dx) * u[i-1] 
        
        # --- Diffusion Contributions (Central) ---
        val_diag  = 1.0 + adv_diag + 2.0 * diff_factor
        val_off_L = adv_off_L - diff_factor
        val_off_R = - diff_factor

        # --- FIX: Handle Periodic Wrapping ---
        # We manually wrap indices because P.setValues uses global indices
        col_left  = (i - 1) % N
        col_right = (i + 1) % N

        rows = [i]
        cols = [col_left, i, col_right]
        vals = [val_off_L, val_diag, val_off_R]
        
        P.setValues(rows, cols, vals, PETSc.InsertMode.INSERT_VALUES)
        
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

# Initial Condition: Top Hat
u_arr = da.getVecArray(u)
xs, xe = da.getRanges()[0]
for i in range(xs, xe):
    x = i * dx
    if 0.1 < x < 0.4:
        u_arr[i] = 1.0
    else:
        u_arr[i] = 0.2
u.copy(prev_u)

# -----------------------------
# 5. Solver Configuration
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
step = 0
total_snes_its = 0
total_ksp_its = 0

if PETSc.COMM_WORLD.getRank() == 0:
    print("-" * 55)
    print(f"{'Time':>10} | {'SNES Its':>10} | {'KSP Its':>10}")
    print("-" * 55)

while t < T_final:
    snes.solve(None, u)
    
    snes_its = snes.getIterationNumber()
    ksp_its = snes.getLinearSolveIterations()
    total_snes_its += snes_its
    total_ksp_its += ksp_its
    
    if PETSc.COMM_WORLD.getRank() == 0:
        print(f"{t:10.3f} | {snes_its:10} | {ksp_its:10}")
        
    u.copy(prev_u)
    t += dt
    step += 1

if PETSc.COMM_WORLD.getRank() == 0:
    print("-" * 55)
    print(f"Total SNES (Newton) Iterations: {total_snes_its}")
    print(f"Total KSP (Linear) Iterations:  {total_ksp_its}")
    print("-" * 55)
    print("Viscous Burgers Simulation Complete.")