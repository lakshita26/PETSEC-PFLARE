import sys
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
import numpy as np

# -----------------------------
# 1. Parameters & Grid Setup
# -----------------------------
N = 256             # Number of grid points
L = 1.0             # Domain length
dx = L / N
dt = 0.005          # Time step (small enough for stability)
T_final = 0.5       # Final simulation time

# Create 1D Distributed Array (DMDA)
# stencil_width=2 is needed for the 2nd neighbor access in MUSCL
da = PETSc.DMDA().create(dim=1, sizes=[N], boundary_type=('periodic',), stencil_width=2)
da.setUp()

def vanleer(r):
    """Van Leer Limiter Function"""
    return (r + abs(r)) / (1.0 + abs(r))

# -----------------------------
# 2. FormFunction (Physics Residual)
# -----------------------------
def FormFunction(snes, U, F_vec):
    # Create local vector to handle ghost points automatically
    U_loc = da.createLocalVec()
    da.globalToLocal(U, U_loc)
    
    u = da.getVecArray(U_loc)
    f = da.getVecArray(F_vec)
    u_old = da.getVecArray(prev_u)
    
    xs, xe = da.getRanges()[0]
    
    for i in range(xs, xe):
        # --- A. Reconstruct value at Right Interface (i + 1/2) ---
        denom_i = (u[i+1] - u[i])
        if abs(denom_i) < 1e-12: 
            r_i = 0.0
        else:
            r_i = (u[i] - u[i-1]) / denom_i
            
        u_face_right = u[i] + 0.5 * vanleer(r_i) * (u[i+1] - u[i])

        # --- B. Reconstruct value at Left Interface (i - 1/2) ---
        denom_im1 = (u[i] - u[i-1])
        if abs(denom_im1) < 1e-12:
            r_im1 = 0.0
        else:
            r_im1 = (u[i-1] - u[i-2]) / denom_im1
            
        u_face_left = u[i-1] + 0.5 * vanleer(r_im1) * (u[i] - u[i-1])

        # --- C. Compute Fluxes (Burgers: F(u) = 0.5 * u^2) ---
        Flux_right = 0.5 * u_face_right**2
        Flux_left  = 0.5 * u_face_left**2
        
        # --- D. Residual Calculation (Backward Euler) ---
        # Res = (u - u_old)/dt + (Flux_right - Flux_left)/dx
        f[i] = (u[i] - u_old[i]) + (dt / dx) * (Flux_right - Flux_left)

# -----------------------------
# 3. FormJacobian (Preconditioner)
# -----------------------------
def FormJacobian(snes, U, J, P):
    """
    Builds a simplified Jacobian matrix for the preconditioner.
    Approximates the system using 1st-order Upwind linearization.
    """
    U_loc = da.createLocalVec()
    da.globalToLocal(U, U_loc)
    u = da.getVecArray(U_loc)
    
    P.zeroEntries()
    xs, xe = da.getRanges()[0]
    
    for i in range(xs, xe):
        # Local wave speed approximation
        a_local = u[i] 
        
        # Simple 1st order upwind Jacobian entries
        # Diagonal term:     1/dt + u_i/dx
        # Off-diagonal term: -u_{i-1}/dx (contribution from left neighbor)
        
        val_diag = 1.0 + (dt / dx) * a_local
        val_off  = -(dt / dx) * u[i-1]
        
        # Insert values into the matrix
        P.setValues([i], [i, i-1], [val_diag, val_off], PETSc.InsertMode.INSERT_VALUES)
        
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

# Initial Condition: Top Hat profile
u_arr = da.getVecArray(u)
xs, xe = da.getRanges()[0]
for i in range(xs, xe):
    x = i * dx
    if 0.1 < x < 0.4:
        u_arr[i] = 1.0
    else:
        u_arr[i] = 0.2 # Background velocity > 0 to ensure upwinding validity
u.copy(prev_u)

# -----------------------------
# 5. Solver Configuration
# -----------------------------
snes = PETSc.SNES().create()
snes.setDM(da)
snes.setFunction(FormFunction, F)
snes.setJacobian(FormJacobian, P_mat)

# Allow command line options (e.g., -snes_monitor -ksp_monitor)
snes.setFromOptions()

# -----------------------------
# 6. Time Loop with Iteration Counting
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
    # Solve the non-linear system for the current time step
    snes.solve(None, u)
    
    # Retrieve iteration counts
    snes_its = snes.getIterationNumber()
    ksp_its = snes.getLinearSolveIterations()
    
    # Update totals
    total_snes_its += snes_its
    total_ksp_its += ksp_its
    
    if PETSc.COMM_WORLD.getRank() == 0:
        print(f"{t:10.3f} | {snes_its:10} | {ksp_its:10}")
        
    # Prepare for next step
    u.copy(prev_u)
    t += dt
    step += 1

if PETSc.COMM_WORLD.getRank() == 0:
    print("-" * 55)
    print(f"Total SNES (Newton) Iterations: {total_snes_its}")
    print(f"Total KSP (Linear) Iterations:  {total_ksp_its}")
    print("-" * 55)
    print("Burgers Simulation Complete.")
    