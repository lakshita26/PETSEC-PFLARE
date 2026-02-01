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
Linear Viscous Advection Equation
Small Viscosity Test
'''

# --------------------------------------------------
# Parameters
# --------------------------------------------------
a = 1.0
nu = 0.0001              # <--- CHANGE: Very small viscosity
n = 100
dx = 1.0 / (n - 1)
dt = 0.01

lam = a * dt / dx        # Courant number (Advection) ~ 1.0
gamma = nu * dt / dx**2  # Diffusion number ~ 0.01 (Now very small)

nt = 100

# --------------------------------------------------
# DMDA grid
# --------------------------------------------------
da = PETSc.DMDA().create(
    sizes=[n],
    dof=1,
    stencil_width=1,
    boundary_type=PETSc.DM.BoundaryType.NONE
)
da.setUniformCoordinates(0.0, 1.0)

# --------------------------------------------------
# Initial condition
# --------------------------------------------------
u_initial = da.createGlobalVec()
(xs, xe) = da.getRanges()[0]

with da.getVecArray(u_initial) as arr:
    i = np.arange(xs, xe)
    x = i * dx
    arr[:] = np.exp(-((x - 0.25) ** 2) / (2 * 0.05 ** 2))

# --------------------------------------------------
# Matrix A assembly
# --------------------------------------------------
A = da.createMatrix()
row = PETSc.Mat.Stencil()
col = PETSc.Mat.Stencil()

for i in range(xs, xe):
    row.index = (i,)

    # Diagonal: 1 + lam + 2*gamma
    # With small nu, gamma is small, so this is closer to (1 + lam)
    col.index = (i,)
    val_diag = 1.0 + lam + 2.0 * gamma
    A.setValueStencil(row, col, val_diag)

    # Left Neighbor (i-1): -lam - gamma
    # Dominated by -lam (advection)
    if i - 1 >= 0:
        col.index = (i - 1,)
        val_left = -lam - gamma
        A.setValueStencil(row, col, val_left)

    # Right Neighbor (i+1): -gamma
    # This will be very close to 0 now!
    if i + 1 < n:
        col.index = (i + 1,)
        val_right = -gamma
        A.setValueStencil(row, col, val_right)

A.assemblyBegin()
A.assemblyEnd()

# --------------------------------------------------
# SHOW MATRIX A
# --------------------------------------------------
A_dense = PETSc.Mat().createDense(size=A.getSize(), comm=A.getComm())
A_dense.setUp()
A_dense.axpy(1.0, A)
A_dense.assemble()
A_np = A_dense.getDenseArray()

print(f"\n===== Matrix A (nu = {nu}) =====")
print(f"Gamma (Diffusion Coeff) is now: {gamma:.5f}")
print("Notice the upper diagonal is non-zero but very small:\n")

# Use higher precision to see the tiny values
np.set_printoptions(precision=4, suppress=True)
print(A_np)

# --------------------------------------------------
# Solver & Time Stepping
# --------------------------------------------------
ksp = PETSc.KSP().create(comm=A.getComm())
ksp.setOperators(A)
ksp.setType(PETSc.KSP.Type.GMRES)
pc = ksp.getPC()
pc.setType(PETSc.PC.Type.JACOBI)
pc.setFromOptions()
ksp.setFromOptions()

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
    ax.set_title(f"Small Viscosity (nu={nu}) | Time = {frame * dt:.2f}")
    return line,

ani = FuncAnimation(fig, update, frames=nt, interval=60, blit=True)
ani.save("1d-advection-upwind-implicit-viscous-small.gif", dpi=150)
plt.close()

print("\nAnimation saved as 1d-advection-upwind-implicit-viscous-small.gif")