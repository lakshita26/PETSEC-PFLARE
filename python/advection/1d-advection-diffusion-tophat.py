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
Linear Viscous Advection Equation (Advection-Diffusion)
u_t + a u_x = nu * u_xx

Implicit Scheme (Backward Euler)
- Advection: Upwind
- Diffusion: Central Difference

Solver: GMRES
Preconditioner: GAMG
'''

# --------------------------------------------------
# Parameters
# --------------------------------------------------
a = 1.0
nu = 0.01
n = 100
dx = 1.0 / (n - 1)
dt = 0.01

lam = a * dt / dx
gamma = nu * dt / dx**2

nt = 100

# --------------------------------------------------
# DMDA grid (DIRICHLET / NON-PERIODIC)
# --------------------------------------------------
da = PETSc.DMDA().create(
    sizes=[n],
    dof=1,
    stencil_width=1,
    boundary_type=PETSc.DM.BoundaryType.NONE
)
da.setUniformCoordinates(0.0, 1.0)

# --------------------------------------------------
# Initial condition (Top Hat)
# --------------------------------------------------
u_initial = da.createGlobalVec()
(xs, xe) = da.getRanges()[0]

with da.getVecArray(u_initial) as arr:
    for i in range(xs, xe):
        x_loc = i * dx
        # Top Hat: u=1 between 0.1 and 0.4, else 0
        if 0.1 <= x_loc <= 0.4:
            arr[i] = 1.0
        else:
            arr[i] = 0.0


# --------------------------------------------------
# Matrix A assembly (IMPLICIT VISCOUS ADVECTION)
# --------------------------------------------------
A = da.createMatrix()
row = PETSc.Mat.Stencil()
col = PETSc.Mat.Stencil()

for i in range(xs, xe):
    row.index = (i,)

    col.index = (i,)
    A.setValueStencil(row, col, 1.0 + lam + 2.0 * gamma)

    if i - 1 >= 0:
        col.index = (i - 1,)
        A.setValueStencil(row, col, -lam - gamma)

    if i + 1 < n:
        col.index = (i + 1,)
        A.setValueStencil(row, col, -gamma)

A.assemblyBegin()
A.assemblyEnd()

# --------------------------------------------------
# Linear solver 
# --------------------------------------------------
ksp = PETSc.KSP().create(comm=A.getComm())
ksp.setOperators(A)
ksp.setType(PETSc.KSP.Type.GMRES)

pc = ksp.getPC()
pc.setType(PETSc.PC.Type.GAMG)

pc.setFromOptions()
ksp.setFromOptions()

# --------------------------------------------------
# Time stepping
# --------------------------------------------------
u = u_initial.copy()
u_new = da.createGlobalVec()
b = da.createGlobalVec()

solution_history = []

for step in range(nt):
    with da.getVecArray(u) as u_arr, da.getVecArray(b) as b_arr:
        b_arr[:] = u_arr[:]

    ksp.solve(b, u_new)
    u.copy(u_new)

    solution_history.append(u.getArray().copy())

print(f"Iterations = {ksp.getIterationNumber()}")
print(f"Converged Reason = {ksp.getConvergedReason()}")
print(f"Final Residual Norm = {ksp.getResidualNorm()}") 


# --------------------------------------------------
# Compute PDE residual norm 
# --------------------------------------------------
R_petsc = da.createGlobalVec()
u_old_temp = da.createGlobalVec()

with da.getVecArray(u) as u_arr, \
     da.getVecArray(u_old_temp) as u_old_arr, \
     da.getVecArray(R_petsc) as R_arr:

    for i in range(xs, xe):
        ut = (u_arr[i] - u_old_arr[i]) / dt

        ux = 0.0
        uxx = 0.0

        if 0 < i < n-1:
            ux  = (u_arr[i] - u_arr[i-1]) / dx   # upwind
            uxx = (u_arr[i+1] - 2*u_arr[i] + u_arr[i-1]) / dx**2

        R_arr[i] = ut + a*ux - nu*uxx

petsc_residual_norm = R_petsc.norm()
print(f"[PETSc] PDE Residual Norm = {petsc_residual_norm:.3e}")


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

def update(frame):
    line.set_ydata(solution_history[frame])
    ax.set_title(f"Implicit Viscous Advection | Time = {frame * dt:.2f}")
    return line,

ani = FuncAnimation(
    fig,
    update,
    frames=nt,
    interval=60,
    blit=True
)

ani.save("implicit_viscous_advection_tophat.gif", dpi=150)
plt.close()

print("\nAnimation saved as implicit_viscous_advection_tophat.gif")