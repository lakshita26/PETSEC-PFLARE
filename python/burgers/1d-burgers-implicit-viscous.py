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
1D Viscous Burgers Equation
u_t + u u_x = nu u_xx

Implicit Backward Euler
Upwind discretization (convection)
Central difference (diffusion)
Newton nonlinear solver (SNES)
GMRES + Jacobi preconditioner
'''

# --------------------------------------------------
# Parameters
# --------------------------------------------------
n = 300                # Grid points
dx = 1.0 / (n - 1)
dt = 0.001             # Time step
nt = 1000              # <--- CHANGED: 1000 steps * 0.001 dt = 1.0 total time
tol_newton = 1e-10
nu = 0.01              # Viscosity coefficient

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
# Vectors & Matrix
# --------------------------------------------------
u = da.createGlobalVec()
u_prev = da.createGlobalVec()
local_u = da.createLocalVec()

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

    da.globalToLocal(x, local_u)

    with da.getVecArray(local_u) as un, \
         da.getVecArray(u_prev) as uold, \
         da.getVecArray(f) as res:

        for i in range(xs, xe):

            # Dirichlet BC
            if i == 0 or i == n - 1:
                res[i] = un[i]
            else:
                ui = un[i]
                uim1 = un[i-1]
                uip1 = un[i+1]

                time_term = (ui - uold[i]) / dt
                convection_term = ui * (ui - uim1) / dx
                diffusion_term = -nu * (uip1 - 2*ui + uim1) / dx**2

                res[i] = time_term + convection_term + diffusion_term

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
                uim1 = un[i-1]

                # Diagonal
                col.index = (i,)
                val_diag = 1.0/dt + (2*ui - uim1)/dx + 2*nu/dx**2
                P.setValueStencil(row, col, val_diag)

                # Left
                col.index = (i-1,)
                val_left = -ui/dx - nu/dx**2
                P.setValueStencil(row, col, val_left)

                # Right
                col.index = (i+1,)
                val_right = -nu/dx**2
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
pc.setType(PETSc.PC.Type.JACOBI)

snes.setFromOptions()

# --------------------------------------------------
# Time Stepping Loop
# --------------------------------------------------
solution_history = []
solution_history.append(u.getArray().copy())

print(f"{'Step':<5} | {'Time':<6} | {'SNES Its':<8} | {'KSP Its':<8} | {'Res Norm':<10}")
print("-" * 55)

for step in range(1, nt + 1):

    u.copy(u_prev)

    snes.solve(None, u)

    snes_its = snes.getIterationNumber()
    ksp_its = snes.getLinearSolveIterations()
    res_norm = snes.getFunctionNorm()

    solution_history.append(u.getArray().copy())

    # Adjusted print frequency so it doesn't spam too much (every 100 steps)
    if step % 100 == 0 or step == 1:
        print(f"{step:<5d} | {step*dt:<6.3f} | {snes_its:<8d} | {ksp_its:<8d} | {res_norm:.2e}")

print("\nFinal SNES iterations :", snes.getIterationNumber())
print("Final KSP iterations  :", snes.getLinearSolveIterations())
print("Final residual norm   :", f"{snes.getFunctionNorm():.3e}")

# --------------------------------------------------
# Animation
# --------------------------------------------------
x_axis = np.linspace(0.0, 1.0, n)
fig, ax = plt.subplots(figsize=(7, 4))
line, = ax.plot(x_axis, solution_history[0], lw=2)

# ax.set_xlim(0, 1)
# ax.set_ylim(0, 1.2)
# ax.set_xlabel("x")
# ax.set_ylabel("u")
# ax.grid(True, alpha=0.3)

# def update(frame):
#     # Skip some frames if animation is too slow (optional, currently showing all)
#     line.set_ydata(solution_history[frame])
#     ax.set_title(f"Viscous Burgers | t = {frame*dt:.3f}")
#     return line,

# # Increased interval slightly or the GIF will be very fast/large
# ani = FuncAnimation(fig, update, frames=nt, interval=20, blit=True)
# ani.save("1d-burgers-implicit-viscous.gif", dpi=150)
# plt.close()

# --------------------------------------------------
# 3D Surface Plot
# --------------------------------------------------
from mpl_toolkits.mplot3d import Axes3D

solution_array = np.array(solution_history)
# This meshgrid will now automatically scale to nt*dt = 1.0
X, T = np.meshgrid(x_axis, np.linspace(0, nt*dt, nt+1))

fig = plt.figure(figsize=(9, 5))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(X, T, solution_array, cmap='viridis', linewidth=0)
ax.set_xlabel("x")
ax.set_ylabel("time")
ax.set_zlabel("u(x,t)")
ax.set_title("Viscous Burgers Equation (Implicit)")

fig.colorbar(surf, shrink=0.5, aspect=12)
plt.tight_layout()
plt.savefig("1d-burgers-implicit-viscous-surface.png", dpi=150)
plt.close()

print("\nAnimation saved as 1d-burgers-implicit-viscous.gif")
print("3D surface plot saved as 1d-burgers-implicit-viscous-surface.png")