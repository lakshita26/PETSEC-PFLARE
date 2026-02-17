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
1D Steady Burgers Equation (viscous)
Equation: u * u_x - nu * u_xx = 0
Method: SNES with Sign-Aware Upwinding for Convection
'''

# --------------------------------------------------
# Parameters
# --------------------------------------------------
n = 200
length = 1.0
nu = 0.01
left_bc = 1.0
right_bc = 0.0

dx = length / (n - 1)

# --------------------------------------------------
# DMDA grid
# --------------------------------------------------
da = PETSc.DMDA().create(
    sizes=[n],
    dof=1,
    stencil_width=1,
    boundary_type=PETSc.DM.BoundaryType.NONE
)

da.setUniformCoordinates(0.0, length)

# --------------------------------------------------
# Vectors & Matrix
# --------------------------------------------------
u = da.createGlobalVec()
local_u = da.createLocalVec()
J = da.createMatrix()

(xs, xe) = da.getRanges()[0]

# --------------------------------------------------
# Initial guess (Smooth transition)
# --------------------------------------------------
with da.getVecArray(u) as arr:
    for i in range(xs, xe):
        x_loc = i * dx
        # Using a linear profile for better steady-state convergence
        arr[i] = left_bc + (right_bc - left_bc) * (x_loc / length)


# --------------------------------------------------
# Residual Function
# --------------------------------------------------

def formResidual(snes, x, f):
    da.globalToLocal(x, local_u)

    with da.getVecArray(local_u) as un, da.getVecArray(f) as res:
        for i in range(xs, xe):
            if i == 0:
                res[i] = un[i] - left_bc
            elif i == n - 1:
                res[i] = un[i] - right_bc
            else:
                ui = un[i]
                uim1 = un[i - 1]
                uip1 = un[i + 1]

                # Diffusion term (Central Difference)
                diffusion = -nu * (uip1 - 2.0 * ui + uim1) / (dx * dx)

                # Convection term (Sign-Aware Upwind)
                # Corrects the instability seen in the previous graph
                if ui > 0:
                    convection = ui * (ui - uim1) / dx
                else:
                    convection = ui * (uip1 - ui) / dx

                res[i] = diffusion + convection

# --------------------------------------------------
# Jacobian Function
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
                uim1 = un[i - 1]
                uip1 = un[i + 1]

                diff_diag = (2.0 * nu) / (dx * dx)
                diff_off  = (-nu) / (dx * dx)

                if ui > 0:
                    conv_diag = (2.0 * ui - uim1) / dx
                    conv_left = -ui / dx
                    conv_right = 0.0
                else:
                    conv_diag = (uip1 - 2.0 * ui) / dx
                    conv_right = ui / dx
                    conv_left = 0.0

                # Set entries
                col.index = (i,)
                P.setValueStencil(row, col, diff_diag + conv_diag)
                col.index = (i - 1,)
                P.setValueStencil(row, col, diff_off + conv_left)
                col.index = (i + 1,)
                P.setValueStencil(row, col, diff_off + conv_right)

    P.assemblyBegin()
    P.assemblyEnd()
    return PETSc.Mat.Structure.SAME_NONZERO_PATTERN

# --------------------------------------------------
# Solver Setup
# --------------------------------------------------

snes = PETSc.SNES().create(comm=da.getComm())
snes.setFunction(formResidual, da.createGlobalVec())
snes.setJacobian(formJacobian, J)

ksp = snes.getKSP()
ksp.setType(PETSc.KSP.Type.GMRES)
pc = ksp.getPC()
pc.setType(PETSc.PC.Type.GAMG)

snes.setFromOptions()

solution_history = []
# Initial guess
solution_history.append(u.getArray().copy())

def snes_monitor(snes, iteration, norm):
    x = snes.getSolution()
    tmp = x.duplicate()
    x.copy(tmp)
    solution_history.append(tmp.getArray().copy())
    print(f"Iter {iteration}: Norm {norm:.2e}")

snes.setMonitor(snes_monitor)

# --------------------------------------------------
# Execution & Plotting
# --------------------------------------------------

print("\nSolving...")
snes.solve(None, u)
print("\nFinal Solver Statistics")
print(f"Final SNES iterations : {snes.getIterationNumber()}")
print(f"Final KSP iterations  : {snes.getLinearSolveIterations()}")
print(f"Final residual norm   : {snes.getFunctionNorm():.3e}")

x_axis = np.linspace(0.0, length, n)
plt.figure(figsize=(7, 4))
plt.plot(x_axis, u.getArray(), lw=2)
plt.xlabel("x")
plt.ylabel("u")
plt.title("1D Steady Burgers (SNES)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("1d-steady-burgers-viscous-implicit.png", dpi=150)
plt.close()

print("\nSaved plot as 1d-steady-burgers-viscous-implicit.png")

if len(solution_history) > 1:
    fig, ax = plt.subplots(figsize=(7, 4))
    line, = ax.plot(x_axis, solution_history[0], lw=2)
    ax.set_xlim(0.0, length)
    ax.set_ylim(-0.2, 1.2)
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.grid(True, alpha=0.3)

    def update(frame):
        line.set_ydata(solution_history[frame])
        ax.set_title(f"1D Steady Burgers (SNES) | iter = {frame}")
        return line,

    ani = FuncAnimation(fig, update, frames=len(solution_history), interval=100, blit=True)
    ani.save("1d-steady-burgers-viscous-implicit.gif", dpi=150)
    plt.close()
    print("Animation saved as 1d-steady-burgers-viscous-implicit.gif")
