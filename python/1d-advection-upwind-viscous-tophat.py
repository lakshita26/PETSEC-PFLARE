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
Preconditioner: AMG (GAMG)
'''

# --------------------------------------------------
# Parameters
# --------------------------------------------------
a = 1.0
nu = 0.01                # <--- CHANGE 1: Added viscosity parameter
n = 100
dx = 1.0 / (n - 1)
dt = 0.01

lam = a * dt / dx        # Courant number (Advection)
gamma = nu * dt / dx**2  # <--- CHANGE 2: Added diffusion number

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
# Discretization:
# (-lam - gamma) * u_{i-1} + (1 + lam + 2*gamma) * u_i + (-gamma) * u_{i+1} = u_i^n

A = da.createMatrix()
row = PETSc.Mat.Stencil()
col = PETSc.Mat.Stencil()

for i in range(xs, xe):
    row.index = (i,)

    # <--- CHANGE 3: Updated Diagonal (Center) Coefficient
    col.index = (i,)
    val_diag = 1.0 + lam + 2.0 * gamma
    A.setValueStencil(row, col, val_diag)

    # <--- CHANGE 4: Updated Left Neighbor (i-1)
    if i - 1 >= 0:
        col.index = (i - 1,)
        val_left = -lam - gamma
        A.setValueStencil(row, col, val_left)

    # <--- CHANGE 5: Added Right Neighbor (i+1) for Diffusion
    if i + 1 < n:
        col.index = (i + 1,)
        val_right = -gamma
        A.setValueStencil(row, col, val_right)

A.assemblyBegin()
A.assemblyEnd()

# --------------------------------------------------
# SHOW MATRIX A (UNCHANGED)
# --------------------------------------------------
print("\n===== Matrix A (PETSc view) =====")
A.view()

A_dense = PETSc.Mat().createDense(size=A.getSize(), comm=A.getComm())
A_dense.setUp()
A_dense.axpy(1.0, A)   # copy values safely
A_dense.assemble()

A_np = A_dense.getDenseArray()

print("\n===== Matrix A (Dense NumPy) =====")
np.set_printoptions(precision=3, suppress=True)
print(A_np)

# --------------------------------------------------
# Linear solver (GMRES + AMG)
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

# --------------------------------------------------
# Animation (numerical solution)
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
    # <--- CHANGE 6: Updated Title
    ax.set_title(f"Implicit Viscous Advection | Time = {frame * dt:.2f}")
    return line,

ani = FuncAnimation(
    fig,
    update,
    frames=nt,
    interval=60,
    blit=True
)

# <--- CHANGE 7: Updated Filename
ani.save("1d-advection-upwind-implicit-viscous-tophat.gif", dpi=150)
plt.close()

print("\nAnimation saved as 1d-advection-upwind-implicit-viscous-tophat.gif")