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

"""
1D Inviscid Burgers Equation
u_t + u u_x = 0

Implicit Backward Euler
Upwind + TVD Minmod limiter
Newton nonlinear solver (SNES)
GMRES + Jacobi preconditioner
"""

# --------------------------------------------------
# Parameters
# --------------------------------------------------
n = 200
dx = 1.0 / (n - 1)
dt = 0.001
nt = 300

# --------------------------------------------------
# Flux limiter (MINMOD)
# --------------------------------------------------
def minmod(a, b):
    return 0.5 * (np.sign(a) + np.sign(b)) * np.minimum(abs(a), abs(b))

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

(xs, xe) = da.getRanges()[0]

# --------------------------------------------------
# Vectors & Matrix
# --------------------------------------------------
u = da.createGlobalVec()
u_prev = da.createGlobalVec()
local_u = da.createLocalVec()
J = da.createMatrix()

# --------------------------------------------------
# Initial condition (Gaussian)
# --------------------------------------------------
with da.getVecArray(u) as arr:
    for i in range(xs, xe):
        x = i * dx
        arr[i] = np.exp(-((x - 0.3)**2) / (2 * 0.05**2))

# --------------------------------------------------
# Residual function with TVD limiter
# --------------------------------------------------
def formResidual(snes, x, f):
    da.globalToLocal(x, local_u)

    with da.getVecArray(local_u) as un, \
         da.getVecArray(u_prev) as uold, \
         da.getVecArray(f) as res:

        for i in range(xs, xe):

            if i == 0 or i == n - 1:
                # Dirichlet BC
                res[i] = un[i]
            else:
                ui   = un[i]
                uim1 = un[i-1]
                uip1 = un[i+1]

                # Backward Euler time term
                time_term = (ui - uold[i]) / dt

                # Slopes
                duL = ui - uim1
                duR = uip1 - ui

                # Minmod limiter
                slope = minmod(duL, duR)

                # Reconstructed upwind state
                u_face = ui - 0.5 * slope

                # Upwind convection
                convection = ui * (u_face - uim1) / dx

                res[i] = time_term + convection

# --------------------------------------------------
# Jacobian (frozen linearization)
# --------------------------------------------------
def formJacobian(snes, x, J, P):
    P.zeroEntries()
    da.globalToLocal(x, local_u)

    row = PETSc.Mat.Stencil()
    col = PETSc.Mat.Stencil()

    with da.getVecArray(local_u) as un:
        for i in range(xs, xe):
            row.index = (i,)

            if i == 0 or i == n - 1:
                P.setValueStencil(row, row, 1.0)
            else:
                ui = un[i]

                # Diagonal
                col.index = (i,)
                P.setValueStencil(row, col, 1.0/dt + ui/dx)

                # Left neighbor
                col.index = (i-1,)
                P.setValueStencil(row, col, -ui/dx)

    P.assemblyBegin()
    P.assemblyEnd()
    return PETSc.Mat.Structure.SAME_NONZERO_PATTERN

# --------------------------------------------------
# SNES solver
# --------------------------------------------------
snes = PETSc.SNES().create(comm=da.getComm())
snes.setFunction(formResidual, da.createGlobalVec())
snes.setJacobian(formJacobian, J)

ksp = snes.getKSP()
ksp.setType(PETSc.KSP.Type.GMRES)
ksp.getPC().setType(PETSc.PC.Type.JACOBI)

snes.setFromOptions()

# --------------------------------------------------
# Time stepping
# --------------------------------------------------
solution_history = [u.getArray().copy()]

print(f"{'Step':<5} {'Time':<6} {'SNES':<6} {'KSP':<6} {'Res':<10}")
print("-"*45)

for step in range(1, nt+1):
    u.copy(u_prev)
    snes.solve(None, u)

    solution_history.append(u.getArray().copy())

    if step % 50 == 0 or step == 1:
        print(f"{step:<5} {step*dt:<6.3f} "
              f"{snes.getIterationNumber():<6} "
              f"{snes.getLinearSolveIterations():<6} "
              f"{snes.getFunctionNorm():.2e}")

# --------------------------------------------------
# Animation
# --------------------------------------------------
x_axis = np.linspace(0.0, 1.0, n)
fig, ax = plt.subplots(figsize=(7,4))
line, = ax.plot(x_axis, solution_history[0], lw=2)

ax.set_xlim(0,1)
ax.set_ylim(0,1.2)
ax.set_xlabel("x")
ax.set_ylabel("u")
ax.grid(alpha=0.3)

def update(frame):
    line.set_ydata(solution_history[frame])
    ax.set_title(f"1D Burgers (Implicit + Minmod)  t={frame*dt:.3f}")
    return line,

ani = FuncAnimation(fig, update, frames=nt, interval=30)
ani.save("1d_burgers_implicit_minmod.gif", dpi=150)
plt.close()

print("\nAnimation saved as 1d_burgers_implicit_minmod.gif")
