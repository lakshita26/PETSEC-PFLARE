import numpy as np
import matplotlib.pyplot as plt
import sys
from matplotlib.animation import FuncAnimation

try:
    import petsc4py
    petsc4py.init(sys.argv)
    from petsc4py import PETSc
except ModuleNotFoundError:
    print("petsc4py not found")
    sys.exit()

'''
1D Burgers Equation
u_t + u u_x = 0

Implicit Backward Euler
Upwind discretization
Newton nonlinear solver (SNES)
GMRES + ILU preconditioner
'''

# --------------------------------------------------
# Parameters
# --------------------------------------------------
n = 200                 # Grid points
dx = 1.0 / (n - 1)
dt = 0.001             # Time step
nt = 300                # Number of time steps
tol_newton = 1e-10

# --------------------------------------------------
# DMDA grid
# --------------------------------------------------
da = PETSc.DMDA().create(
    sizes=[n],
    dof=1,
    stencil_width=1,  # Needed for upwind (i-1)
    boundary_type=PETSc.DM.BoundaryType.NONE
)
da.setUniformCoordinates(0.0, 1.0)

# --------------------------------------------------
# Vectors & Matrix
# --------------------------------------------------
u = da.createGlobalVec()        # Current solution
u_prev = da.createGlobalVec()   # Previous time step solution
local_u = da.createLocalVec()   # Local vector (with ghosts) for stencils

# Create Matrix once (memory optimization)
J = da.createMatrix()

# --------------------------------------------------
# Initial condition (Gaussian)
# --------------------------------------------------
(xs, xe) = da.getRanges()[0]
with da.getVecArray(u) as arr:
    for i in range(xs, xe):
        x_loc = i * dx
        arr[i] = np.exp(-((x_loc - 0.3)**2) / (2 * 0.05**2))

# --------------------------------------------------
# Residual Function
# --------------------------------------------------
def formResidual(snes, x, f):
    """
    Calculates F(u) = (u - u_prev)/dt + u * (u - u_left)/dx
    Uses a local vector scatter to handle Ghost Points and Locked Vectors.
    """
    # 1. Scatter Global x -> Local x (Get Ghost Points & Unlock Read Access)
    da.globalToLocal(x, local_u)

    # 2. Compute Residual
    with da.getVecArray(local_u) as un, \
         da.getVecArray(u_prev) as uold, \
         da.getVecArray(f) as res:

        for i in range(xs, xe):
            # Dirichlet BCs: u=0 at boundaries (indexes 0 and n-1)
            if i == 0 or i == n - 1:
                res[i] = un[i] - 0.0
            else:
                ui = un[i]
                uim1 = un[i-1]  # Access ghost point safely

                # Backward Euler Time Term
                time_term = (ui - uold[i]) / dt
                
                # Upwind Convection Term
                convection_term = ui * (ui - uim1) / dx

                res[i] = time_term + convection_term

# --------------------------------------------------
# Jacobian Function
# --------------------------------------------------
def formJacobian(snes, x, J, P):
    """
    Calculates the Jacobian Matrix analytically.
    """
    # 1. Zero out the matrix to start fresh
    P.zeroEntries()
    
    # 2. Scatter Global -> Local for stencil access
    da.globalToLocal(x, local_u)

    row = PETSc.Mat.Stencil()
    col = PETSc.Mat.Stencil()

    with da.getVecArray(local_u) as un:
        for i in range(xs, xe):
            row.index = (i,)
            
            # Boundary Conditions: Identity Row
            if i == 0 or i == n - 1:
                P.setValueStencil(row, row, 1.0)
            else:
                ui = un[i]
                uim1 = un[i-1]

                # Diagonal Entry: derivative w.r.t u[i]
                # F_i = (u_i - u_old)/dt + u_i(u_i - u_i-1)/dx
                # dF/du_i = 1/dt + (2*u_i - u_i-1)/dx
                col.index = (i,)
                val_diag = 1.0/dt + (2*ui - uim1)/dx
                P.setValueStencil(row, col, val_diag)

                # Off-Diagonal Entry (Left): derivative w.r.t u[i-1]
                # dF/du_i-1 = -u_i/dx
                col.index = (i-1,)
                val_left = -ui/dx
                P.setValueStencil(row, col, val_left)

    # 3. Assemble Matrix
    P.assemblyBegin()
    P.assemblyEnd()
    
    return PETSc.Mat.Structure.SAME_NONZERO_PATTERN

# --------------------------------------------------
# Solver Setup (SNES)
# --------------------------------------------------
snes = PETSc.SNES().create(comm=da.getComm())
snes.setFunction(formResidual, da.createGlobalVec())
snes.setJacobian(formJacobian, J)

# Configure KSP (Linear Solver) inside SNES
ksp = snes.getKSP()
ksp.setType(PETSc.KSP.Type.GMRES)
pc = ksp.getPC()
pc.setType(PETSc.PC.Type.JACOBI) # ILU is often better for transport/convection

# Allow command line overrides (e.g. -snes_monitor -ksp_monitor)
snes.setFromOptions()

# --------------------------------------------------
# Time Stepping Loop
# --------------------------------------------------
solution_history = []
solution_history.append(u.getArray().copy())

print(f"{'Step':<5} | {'Time':<6} | {'SNES Its':<8} | {'KSP Its':<8} | {'Res Norm':<10}")
print("-" * 55)

for step in range(1, nt + 1):
    
    # Copy current u to u_prev
    u.copy(u_prev)

    # Solve nonlinear system: F(u) = 0
    snes.solve(None, u)

    # Get Iteration Counts
    snes_its = snes.getIterationNumber()
    ksp_its = snes.getLinearSolveIterations()
    res_norm = snes.getFunctionNorm()

    # Store history
    solution_history.append(u.getArray().copy())

    # Logging
    if step % 50 == 0 or step == 1:
        print(f"{step:<5d} | {step*dt:<6.3f} | {snes_its:<8d} | {ksp_its:<8d} | {res_norm:.2e}")

    # Print Jacobian Matrix at a specific step (e.g., step 50)
    if step == 50:
        print("\n" + "="*30)
        print(f" Jacobian Matrix A at Step {step}")
        print("="*30)
        # We need to refresh the Jacobian view from the solver
        # Note: 'J' is already updated inside SNES
        J.view() 
        print("="*30 + "\n")



print("\n" + "="*40)
print(" Final Solver Statistics")
print("="*40)
print(f"Final SNES iterations : {snes.getIterationNumber()}")
print(f"Final KSP iterations  : {snes.getLinearSolveIterations()}")
print(f"Final residual norm   : {snes.getFunctionNorm():.3e}")
print("="*40)

    
# --------------------------------------------------
# Animation
# --------------------------------------------------
x_axis = np.linspace(0.0, 1.0, n)
fig, ax = plt.subplots(figsize=(7, 4))
line, = ax.plot(x_axis, solution_history[0], lw=2)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1.2)
ax.set_xlabel("x")
ax.set_ylabel("u")
ax.grid(True, alpha=0.3)

def update(frame):
    line.set_ydata(solution_history[frame])
    ax.set_title(f"Burgers (SNES) | t = {frame*dt:.3f}")
    return line,

ani = FuncAnimation(fig, update, frames=nt, interval=30, blit=True)
ani.save("1d-burgers-implicit.gif", dpi=150)
plt.close()

print("\nAnimation saved as 1d-burgers-implicit.gif")