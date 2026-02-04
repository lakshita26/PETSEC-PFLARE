import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import cm
import sys

try:
    import petsc4py
    petsc4py.init(sys.argv)
    from petsc4py import PETSc
except ModuleNotFoundError:
    print("petsc4py not found")
    sys.exit()

'''
2D Inviscid Burgers Equation
u_t + u u_x + u u_y = 0

Implicit Backward Euler
Upwind discretization
Newton nonlinear solver (SNES)
GMRES + Jacobi
'''

# --------------------------------------------------
# Parameters
# --------------------------------------------------
nx = 120
ny = 120

dx = 1.0 / (nx - 1)
dy = 1.0 / (ny - 1)

dt = 0.002
nt = 200

# --------------------------------------------------
# DMDA grid
# --------------------------------------------------
da = PETSc.DMDA().create(
    dim=2,
    sizes=[nx, ny],
    dof=1,
    stencil_width=1,
    boundary_type=PETSc.DM.BoundaryType.NONE
)
da.setUniformCoordinates(0.0, 1.0, 0.0, 1.0)

# --------------------------------------------------
# Vectors & Matrix
# --------------------------------------------------
u = da.createGlobalVec()
u_prev = da.createGlobalVec()
local_u = da.createLocalVec()
J = da.createMatrix()

(xs, xe), (ys, ye) = da.getRanges()

# --------------------------------------------------
# Initial condition (Top Hat)
# --------------------------------------------------
u_initial = da.createGlobalVec()
(xs, xe), (ys, ye) = da.getRanges()

with da.getVecArray(u_initial) as arr:
    for j in range(ys, ye):
        for i in range(xs, xe):
            x = i * dx
            y = j * dy
            
            if abs(x) < 0.5 and abs(y) < 0.5:
                arr[j, i] = 1.0   # High
            else:
                arr[j, i] = 0.0   # Low

# --------------------------------------------------
# Residual
# --------------------------------------------------
def formResidual(snes, x, f):
    da.globalToLocal(x, local_u)

    with da.getVecArray(local_u) as un, \
         da.getVecArray(u_prev) as uold, \
         da.getVecArray(f) as res:

        for j in range(ys, ye):
            for i in range(xs, xe):

                if i == 0 or j == 0 or i == nx-1 or j == ny-1:
                    res[j, i] = un[j, i]
                else:
                    ui = un[j, i]
                    uim1 = un[j, i-1]
                    ujm1 = un[j-1, i]

                    time_term = (ui - uold[j, i]) / dt
                    conv_x = ui * (ui - uim1) / dx
                    conv_y = ui * (ui - ujm1) / dy

                    res[j, i] = time_term + conv_x + conv_y

# --------------------------------------------------
# Jacobian
# --------------------------------------------------
def formJacobian(snes, x, J, P):
    P.zeroEntries()
    da.globalToLocal(x, local_u)

    row = PETSc.Mat.Stencil()
    col = PETSc.Mat.Stencil()

    with da.getVecArray(local_u) as un:
        for j in range(ys, ye):
            for i in range(xs, xe):
                row.index = (i, j)

                if i == 0 or j == 0 or i == nx-1 or j == ny-1:
                    P.setValueStencil(row, row, 1.0)
                else:
                    ui = un[j, i]
                    uim1 = un[j, i-1]
                    ujm1 = un[j-1, i]

                    diag = (1.0/dt
                            + (2*ui - uim1)/dx
                            + (2*ui - ujm1)/dy)

                    col.index = (i, j)
                    P.setValueStencil(row, col, diag)

                    col.index = (i-1, j)
                    P.setValueStencil(row, col, -ui/dx)

                    col.index = (i, j-1)
                    P.setValueStencil(row, col, -ui/dy)

    P.assemblyBegin()
    P.assemblyEnd()
    return PETSc.Mat.Structure.SAME_NONZERO_PATTERN

# --------------------------------------------------
# SNES Solver
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
# Time Loop
# --------------------------------------------------
solution_history = []
slice_history = []
velocities = []
times = []

mid_y = ny // 2
prev_centroid = None

for step in range(nt):
    u.copy(u_prev)
    snes.solve(None, u)

    sol = u.getArray().reshape(ny, nx).copy()
    solution_history.append(sol)
    slice_history.append(sol[mid_y, :])
    times.append(step * dt)

    mass = np.sum(sol)
    if mass > 1e-10:
        Y, X = np.indices((ny, nx))
        cx = np.sum(X * sol) / mass * dx
        cy = np.sum(Y * sol) / mass * dy
        c = np.array([cx, cy])

        if prev_centroid is not None:
            velocities.append(np.linalg.norm(c - prev_centroid) / dt)
        else:
            velocities.append(0.0)
        prev_centroid = c
    else:
        velocities.append(0.0)

# --------------------------------------------------
# 2D Animation
# --------------------------------------------------
fig, ax = plt.subplots(figsize=(6,5))
im = ax.imshow(solution_history[0], origin='lower',
               extent=[0,1,0,1], cmap='viridis')
fig.colorbar(im)

def update(frame):
    im.set_data(solution_history[frame])
    ax.set_title(f"2D Burgers | t={frame*dt:.3f}")
    return im,

ani = FuncAnimation(fig, update, frames=nt, interval=40)
ani.save("2d-burgers-implicit-tophat.gif", dpi=120)
plt.close()

# --------------------------------------------------
# Space–Time Surface
# --------------------------------------------------
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')

x_vals = np.linspace(0,1,nx)
t_vals = np.array(times)
X, T = np.meshgrid(x_vals, t_vals)
Z = np.array(slice_history)

surf = ax.plot_surface(X, T, Z, cmap=cm.viridis)
ax.set_xlabel("x")
ax.set_ylabel("t")
ax.set_zlabel("u")
fig.colorbar(surf)

plt.savefig("2d-burgers-implicit-tophat-surface.png", dpi=150)
plt.close()

# --------------------------------------------------
# 1D Slicing
# --------------------------------------------------
fig, ax = plt.subplots(figsize=(8,5))
for idx in np.linspace(0, nt-1, 5, dtype=int):
    ax.plot(x_vals, slice_history[idx], label=f"t={idx*dt:.2f}")

ax.set_title("1D Mid-Y Slices (2D Burgers)")
ax.legend()
ax.grid(alpha=0.3)
plt.savefig("2d-burgers-implicit-tophat-1d-slicing.png", dpi=150)
plt.close()

# --------------------------------------------------
# Speed vs Time
# --------------------------------------------------
fig, ax = plt.subplots(figsize=(8,5))
ax.plot(times, velocities, lw=2)
ax.set_xlabel("Time")
ax.set_ylabel("Speed")
ax.set_title("Centroid Speed vs Time")
ax.grid(alpha=0.3)

plt.savefig("2d-burgers-implicit-tophat-ut.png", dpi=150)
plt.close()
