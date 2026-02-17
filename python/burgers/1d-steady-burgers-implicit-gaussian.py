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
1D Steady Burgers Equation

u * u_x = 0 (inviscid)

Steady nonlinear solve (SNES)
Upwind for convection, central for diffusion
'''

# --------------------------------------------------
# Parameters
# --------------------------------------------------
n = 200
length = 1.0
nu = 0.00
left_bc = 1.0
right_bc = 0.0

dx = length / (n - 1)

# --------------------------------------------------
# DMDA grid
# --------------------------------------------------
da = PETSc.DMDA().create(
    sizes=[n],
    dof=1,
    stencil_width=1,         # Needed for upwind (i-1)
    boundary_type=PETSc.DM.BoundaryType.NONE
)

da.setUniformCoordinates(0.0, length)

# --------------------------------------------------
# Vectors & Matrix
# --------------------------------------------------
u = da.createGlobalVec()        # Current solution
local_u = da.createLocalVec()   # Local vector (with ghosts) for stencils
J = da.createMatrix()

(xs, xe) = da.getRanges()[0]

# --------------------------------------------------
# Initial guess (linear profile between BCs)
# --------------------------------------------------
# with da.getVecArray(u) as arr:
#     for i in range(xs, xe):
#         x_loc = i * dx
#         arr[i] = left_bc + (right_bc - left_bc) * (x_loc / length)

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
    Calculates F(u) = -nu * u_xx + u * u_x
    using upwind for u * u_x and central for u_xx.
    """
    da.globalToLocal(x, local_u)

    with da.getVecArray(local_u) as un, da.getVecArray(f) as res:
        for i in range(xs, xe):
            if i == 0:
                res[i] = un[i] - left_bc
            elif i == n - 1:
                # Outflow BC for inviscid case: zero gradient
                res[i] = un[i] - un[i - 1]
            else:
                ui = un[i]
                uim1 = un[i - 1]
                uip1 = un[i + 1]

                # Diffusion term (central)
                diffusion = -nu * (uip1 - 2.0 * ui + uim1) / (dx * dx)

                # Convection term (upwind)
                convection = ui * (ui - uim1) / dx

                res[i] = diffusion + convection

# --------------------------------------------------
# Jacobian Function
# --------------------------------------------------

def formJacobian(snes, x, J, P):
    """
    Analytical Jacobian for steady viscous Burgers.
    """
    P.zeroEntries()
    da.globalToLocal(x, local_u)

    row = PETSc.Mat.Stencil()
    col = PETSc.Mat.Stencil()

    with da.getVecArray(local_u) as un:
        for i in range(xs, xe):
            row.index = (i,)

            if i == 0:
                P.setValueStencil(row, row, 1.0)
            elif i == n - 1:
                # d(u_n - u_{n-1})/d u_n = 1, d/d u_{n-1} = -1
                P.setValueStencil(row, row, 1.0)
                col.index = (i - 1,)
                P.setValueStencil(row, col, -1.0)
            else:
                ui = un[i]

                # d/d u_i
                col.index = (i,)
                val_diag = (2.0 * nu) / (dx * dx) + (2.0 * ui - un[i - 1]) / dx
                P.setValueStencil(row, col, val_diag)

                # d/d u_{i-1}
                col.index = (i - 1,)
                val_left = (-nu) / (dx * dx) - ui / dx
                P.setValueStencil(row, col, val_left)

                # d/d u_{i+1}
                col.index = (i + 1,)
                val_right = (-nu) / (dx * dx)
                P.setValueStencil(row, col, val_right)

    P.assemblyBegin()
    P.assemblyEnd()

    return PETSc.Mat.Structure.SAME_NONZERO_PATTERN

# --------------------------------------------------
# Solver Setup (SNES)
# --------------------------------------------------

snes = PETSc.SNES().create(comm=da.getComm())
snes.setFunction(formResidual, da.createGlobalVec())
snes.setJacobian(formJacobian, J)

ksp = snes.getKSP()
ksp.setType(PETSc.KSP.Type.GMRES)
pc = ksp.getPC()
# pc.setType(PETSc.PC.Type.ILU)
# pc.setType(PETSc.PC.Type.GAMG)
pc.setType(PETSc.PC.Type.JACOBI)

snes.setFromOptions()

# --------------------------------------------------
# Solve (with monitor to capture iterations)
# --------------------------------------------------

solution_history = []
solution_history.append(u.getArray().copy())

def snes_monitor(snes, iteration, norm):
    x = snes.getSolution()
    tmp = x.duplicate()
    x.copy(tmp)
    solution_history.append(tmp.getArray().copy())

snes.setMonitor(snes_monitor)

print(f"{'Step':<5} | {'Time':<6} | {'SNES Its':<8} | {'KSP Its':<8} | {'Res Norm':<10}")
print("-" * 55)

snes.solve(None, u)

print(f"{1:<5d} | {0.0:<6.3f} | {snes.getIterationNumber():<8d} | {snes.getLinearSolveIterations():<8d} | {snes.getFunctionNorm():.2e}")

print("\nFinal Solver Statistics")
print(f"Final SNES iterations : {snes.getIterationNumber()}")
print(f"Final KSP iterations  : {snes.getLinearSolveIterations()}")
print(f"Final residual norm   : {snes.getFunctionNorm():.3e}")

# --------------------------------------------------
# Plot
# --------------------------------------------------

x_axis = np.linspace(0.0, length, n)
plt.figure(figsize=(7, 4))
plt.plot(x_axis, u.getArray(), lw=2)
plt.xlabel("x")
plt.ylabel("u")
plt.title("1D Steady Burgers (SNES)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("1d-steady-burgers-gaussian.png", dpi=150)
plt.close()

print("\nSaved plot as 1d-steady-burgers-gaussian.png")

# --------------------------------------------------
# Animation (SNES iterations)
# --------------------------------------------------

if len(solution_history) > 1:
    fig, ax = plt.subplots(figsize=(7, 4))
    line, = ax.plot(x_axis, solution_history[0], lw=2)
    ax.set_xlim(0.0, length)
    ax.set_ylim(min(0.0, right_bc) - 0.2, max(left_bc, 1.0) + 0.2)
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.grid(True, alpha=0.3)

    def update(frame):
        line.set_ydata(solution_history[frame])
        ax.set_title(f"1D Steady Burgers (SNES) | iter = {frame}")
        return line,

    ani = FuncAnimation(fig, update, frames=len(solution_history), interval=100, blit=True)
    ani.save("1d-steady-burgers-gaussian.gif", dpi=150)
    plt.close()
    print("Animation saved as 1d-steady-burgers-gaussian.gif")
