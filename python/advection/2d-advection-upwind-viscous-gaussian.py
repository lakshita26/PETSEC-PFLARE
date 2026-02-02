import matplotlib
# Force matplotlib to not use any Xwindow backend
matplotlib.use('Agg') 

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec  
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
2D Linear viscous advection equation
u_t + a_x u_x + a_y u_y = nu (u_xx + u_yy)

Implicit UPWIND + diffusion
Dirichlet boundary condition
Gaussian initial condition
'''

# --------------------------------------------------
# Parameters
# --------------------------------------------------
ax_vel = 1.0
ay_vel = 1.0
nu = 0.01     # <<< ADDED (viscosity)

# Grid size
nx = 120
ny = 120

dx = 1.0 / (nx - 1)
dy = 1.0 / (ny - 1)

dt = 0.01

# CFL numbers (advection)
lam_x = ax_vel * dt / dx
lam_y = ay_vel * dt / dy

# CFL numbers (diffusion)   <<< ADDED
mu_x = nu * dt / dx**2
mu_y = nu * dt / dy**2

nt = 100

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
da.setUniformCoordinates(xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0)

# --------------------------------------------------
# Initial condition (2D Gaussian)
# --------------------------------------------------
u_initial = da.createGlobalVec()
(xs, xe), (ys, ye) = da.getRanges()

with da.getVecArray(u_initial) as arr:
    for j in range(ys, ye):
        for i in range(xs, xe):
            x = i * dx
            y = j * dy
            arr[j, i] = np.exp(-((x - 0.2)**2 + (y - 0.2)**2) / (2 * 0.05**2))

# --------------------------------------------------
# Matrix A assembly (IMPLICIT UPWIND + DIFFUSION)
# --------------------------------------------------
A = da.createMatrix()
row = PETSc.Mat.Stencil()
col = PETSc.Mat.Stencil()

diag_val = 1.0 + lam_x + lam_y + 2.0 * (mu_x + mu_y)   # <<< MODIFIED

for j in range(ys, ye):
    for i in range(xs, xe):
        row.index = (i, j)

        col.index = (i, j)
        A.setValueStencil(row, col, diag_val)

        if i - 1 >= 0:
            col.index = (i - 1, j)
            A.setValueStencil(row, col, -(lam_x + mu_x))  # <<< MODIFIED
            
        if i + 1 < nx:
            col.index = (i + 1, j)
            A.setValueStencil(row, col, -mu_x)            # <<< ADDED

        if j - 1 >= 0:
            col.index = (i, j - 1)
            A.setValueStencil(row, col, -(lam_y + mu_y))  # <<< MODIFIED
            
        if j + 1 < ny:
            col.index = (i, j + 1)
            A.setValueStencil(row, col, -mu_y)            # <<< ADDED

A.assemblyBegin()
A.assemblyEnd()

# --------------------------------------------------
# Linear solver (UNCHANGED)
# --------------------------------------------------
ksp = PETSc.KSP().create(comm=A.getComm())
ksp.setOperators(A)
ksp.setType(PETSc.KSP.Type.GMRES)
pc = ksp.getPC()
pc.setType(PETSc.PC.Type.GAMG)
pc.setFromOptions()
ksp.setFromOptions()

# --------------------------------------------------
# Time stepping & Data Collection (UNCHANGED)
# --------------------------------------------------
u = u_initial.copy()
u_new = da.createGlobalVec()
b = da.createGlobalVec()

solution_history = []  
slice_history = []     
velocities = []        
times = []

mid_y = ny // 2
prev_centroid = None

print("Starting Time Loop...")

for step in range(nt):
    u.copy(b)
    ksp.solve(b, u_new)
    u_new.copy(u)
    
    current_time = step * dt
    times.append(current_time)
    
    sol_array = u.getArray().reshape(ny, nx).copy()
    solution_history.append(sol_array)
    slice_history.append(sol_array[mid_y, :])
    
    total_mass = np.sum(sol_array)
    if total_mass > 1e-10:
        Y_grid, X_grid = np.indices((ny, nx))
        x_cm = np.sum(X_grid * sol_array) / total_mass * dx
        y_cm = np.sum(Y_grid * sol_array) / total_mass * dy
        current_centroid = np.array([x_cm, y_cm])
        
        if prev_centroid is not None:
            dist = np.linalg.norm(current_centroid - prev_centroid)
            velocities.append(dist / dt)
        else:
            velocities.append(np.sqrt(ax_vel**2 + ay_vel**2))
        prev_centroid = current_centroid
    else:
        velocities.append(0.0)

print(f"Iterations (Last Step) = {ksp.getIterationNumber()}")

# --------------------------------------------------
# 1. Animation (UNCHANGED)
# --------------------------------------------------
fig_anim, ax_anim = plt.subplots(figsize=(6, 5))
im = ax_anim.imshow(solution_history[0], origin='lower', extent=[0, 1, 0, 1], 
                    cmap='viridis', vmin=0, vmax=1.0)
fig_anim.colorbar(im, ax=ax_anim, label='u')
ax_anim.set_xlabel("x")
ax_anim.set_ylabel("y")
ax_anim.set_title("2D Viscous Advection Animation")

def update(frame):
    im.set_data(solution_history[frame])
    ax_anim.set_title(f"Time = {frame * dt:.2f}")
    return im,

ani = FuncAnimation(fig_anim, update, frames=nt, interval=50, blit=False)
ani.save("2d-viscous-advection-gaussian-heatmap.gif", dpi=100)
plt.close(fig_anim)

# --------------------------------------------------
# 2. Space-Time Graph (UNCHANGED)
# --------------------------------------------------
fig_st = plt.figure(figsize=(10, 7))
ax_st = fig_st.add_subplot(111, projection='3d')

x_vals = np.linspace(0, 1, nx)
t_vals = np.linspace(0, nt*dt, nt)
X_mesh, T_mesh = np.meshgrid(x_vals, t_vals)
Z_mesh = np.array(slice_history)

surf = ax_st.plot_surface(X_mesh, T_mesh, Z_mesh, cmap=cm.viridis)

ax_st.set_xlabel('Space (x)')
ax_st.set_ylabel('Time (t)')
ax_st.set_zlabel('u')
ax_st.set_title(f'Space-Time Evolution (Slice at y={mid_y*dy:.2f})')
fig_st.colorbar(surf, shrink=0.5, aspect=5)

plt.savefig("2d-viscous-advection-gaussian-surface.png", dpi=150)
plt.close(fig_st)

# --------------------------------------------------
# 3. 1D Slicing Snapshots (UNCHANGED)
# --------------------------------------------------
fig_slice, ax_slice = plt.subplots(figsize=(8, 5))
indices_to_plot = np.linspace(0, nt-1, 5, dtype=int)

for idx in indices_to_plot:
    t_label = idx * dt
    ax_slice.plot(x_vals, slice_history[idx], lw=2, label=f't={t_label:.2f}')

ax_slice.set_title("1D Slicing Snapshots (Profile at mid-Y)")
ax_slice.set_xlabel("Space (x)")
ax_slice.set_ylabel("u")
ax_slice.legend()
ax_slice.grid(True, alpha=0.3)

plt.savefig("2d-viscous-advection-gaussian-1d-slicing.png", dpi=150)
plt.close(fig_slice)

# --------------------------------------------------
# 4. Speed vs Time Graph (UNCHANGED)
# --------------------------------------------------
fig_speed, ax_speed = plt.subplots(figsize=(8, 5))
ax_speed.plot(times, velocities, 'r-', lw=2, marker='o', markersize=3)

ax_speed.set_title("Wave Speed vs Time (Centroid Tracking)")
ax_speed.set_xlabel("Time (t)")
ax_speed.set_ylabel("Speed |v|")
ax_speed.grid(True, alpha=0.3)

analytic_speed = np.sqrt(ax_vel**2 + ay_vel**2)
ax_speed.axhline(analytic_speed, color='k', linestyle='--',
                 label=f'Analytic Speed ({analytic_speed:.2f})')
ax_speed.legend()

plt.savefig("2d-viscous-advection-gaussian-ut.png", dpi=150)
plt.close(fig_speed)
