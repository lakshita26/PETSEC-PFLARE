import sys
import petsc4py
petsc4py.init(sys.argv)

from petsc4py import PETSc
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import warnings
warnings.filterwarnings("ignore")

# ======================================================
# 1. PARAMETERS
# ======================================================
N  = 129
dt = 0.002
T  = 1.0  

da = PETSc.DMDA().create(
    dim=2,
    sizes=[N, N],
    dof=2,
    stencil_width=1,
    boundary_type=(PETSc.DM.BoundaryType.GHOSTED, PETSc.DM.BoundaryType.GHOSTED),
    comm=PETSc.COMM_WORLD
)
da.setUniformCoordinates(-1, 1, -1, 1)
da.setUp()

hx = 2.0 / (N - 1)
hy = 2.0 / (N - 1)

# ======================================================
# 2. RHS FUNCTION (ADVECTION / UPWIND)
# ======================================================
def RHSFunction(ts, t, X, F):
    dm = ts.getDM()
    Xloc = dm.createLocalVec()
    dm.globalToLocal(X, Xloc)
    u_arr = dm.getVecArray(Xloc)
    f_arr = dm.getVecArray(F)
    (xs, xe), (ys, ye) = dm.getRanges()

    for j in range(ys, ye):
        for i in range(xs, xe):
            if i == 0 or j == 0 or i == N-1 or j == N-1:
                f_arr[i, j] = (0.0, 0.0)
                continue

            ui, vi = u_arr[i, j]
            # Upwind X
            du_dx = (ui - u_arr[i-1, j][0])/hx if ui >= 0 else (u_arr[i+1, j][0] - ui)/hx
            dv_dx = (vi - u_arr[i-1, j][1])/hx if ui >= 0 else (u_arr[i+1, j][1] - vi)/hx
            # Upwind Y
            du_dy = (ui - u_arr[i, j-1][0])/hy if vi >= 0 else (u_arr[i, j+1][0] - ui)/hy
            dv_dy = (vi - u_arr[i, j-1][1])/hy if vi >= 0 else (u_arr[i, j+1][1] - vi)/hy

            f_arr[i, j][0] = -(ui * du_dx + vi * du_dy)
            f_arr[i, j][1] = -(ui * dv_dx + vi * dv_dy)

# ======================================================
# 3. TS SETUP & IC
# ======================================================
ts = PETSc.TS().create()
ts.setDM(da)
ts.setType(PETSc.TS.Type.SSP)
ts.setRHSFunction(RHSFunction)
ts.setTimeStep(dt)
ts.setMaxTime(T)
ts.setExactFinalTime(PETSc.TS.ExactFinalTime.MATCHSTEP)

X = da.createGlobalVec()
with da.getVecArray(X) as arr:
    (xs, xe), (ys, ye) = da.getRanges()
    for j in range(ys, ye):
        for i in range(xs, xe):
            xi = -1.0 + i * hx
            yi = -1.0 + j * hy
            g = 1.5 * np.exp(-((xi+0.2)**2 + (yi+0.2)**2) / (2*0.15**2))
            arr[i, j] = (g, g)

# ======================================================
# 4. MONITORING
# ======================================================
history_time = []
history_slices = []
history_speed = []

def monitor(ts, step, time, Xvec):
    if PETSc.COMM_WORLD.getRank() == 0:
        A = Xvec.getArray(readonly=True).reshape((N, N, 2))
        u_vals = A[:, :, 0]
        history_time.append(time)
        history_slices.append(u_vals[:, N//2].copy())
        
        # Calculate Speed (Centroid Velocity)
        mass = np.sum(u_vals)
        if mass > 1e-9:
            indices = np.indices((N, N))
            cx = np.sum(indices[1] * u_vals) / mass
            cy = np.sum(indices[0] * u_vals) / mass
            history_speed.append(np.sqrt(cx**2 + cy**2) * hx)
        else:
            history_speed.append(0.0)

ts.setMonitor(monitor)
ts.solve(X)

# ======================================================
# 5. FINAL PLOTS (SAME FILENAMES)
# ======================================================
if PETSc.COMM_WORLD.getRank() == 0:
    U_final = X.getArray().reshape((N, N, 2))[:, :, 0]
    grid = np.linspace(-1, 1, N)
    Xg, Yg = np.meshgrid(grid, grid)

    # Plot 1: 1D Slicing
    plt.figure(figsize=(8,5))
    for idx in [0, len(history_slices)//2, -1]:
        plt.plot(grid, history_slices[idx], label=f"t={history_time[idx]:.2f}")
    plt.title("1D Mid-Y Slices (2D Burgers)")
    plt.legend(); plt.grid(True); plt.savefig("2d-burgers-gaussian-1d-slice.png")

    # Plot 2: 2D Contour
    plt.figure(figsize=(7,6))
    plt.contourf(Xg, Yg, U_final.T, 50, cmap="viridis")
    plt.colorbar(); plt.title("2D Inviscid Burgers Contour"); plt.savefig("2d-burgers-gaussian-2d-contour.png")

    # Plot 3: 3D Surface
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(Xg, Yg, U_final.T, cmap=cm.RdBu_r, antialiased=False)
    ax.set_title("3D Surface (Gaussian IC)"); plt.savefig("2d-burgers-gaussian-surface.png")

    # Plot 4: Space-Time Surface (x vs t)
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection='3d')
    T_mesh, X_mesh = np.meshgrid(np.array(history_time), grid)
    ax.plot_surface(X_mesh, T_mesh, np.array(history_slices).T, cmap='viridis')
    ax.set_xlabel("x"); ax.set_ylabel("time"); ax.set_zlabel("u")
    plt.title("Space-Time Surface Plot"); plt.savefig("2d-burgers-implicit-surface.png")

    # Plot 5: Speed vs Time
    plt.figure(figsize=(8,5))
    plt.plot(history_time, history_speed, lw=2)
    plt.xlabel("Time"); plt.ylabel("Centroid Speed"); plt.title("Centroid Speed vs Time")
    plt.grid(True); plt.savefig("2d-burgers-implicit-ut.png")

    print("\n--- Solver Info ---")
    print(f"Final TS Step: {ts.getStepNumber()}")
    print(f"Final Residual: {ts.getSNES().getFunctionNorm():.4e}")