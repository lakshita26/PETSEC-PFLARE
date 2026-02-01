import matplotlib
# Force matplotlib to not use any Xwindow backend (fixes VS Code display issues)
matplotlib.use('Agg') 

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation
import sys

try:
    import petsc4py
    petsc4py.init(sys.argv)
    from petsc4py import PETSc
except ModuleNotFoundError:
    print("petsc4py not found")
    sys.exit()

'''
2D Linear pure advection equation
u_t + a_x u_x + a_y u_y = 0

Implicit UPWIND scheme
Dirichlet boundary condition
Top Hat / Square Wave Initial Condition
'''

# --------------------------------------------------
# Parameters
# --------------------------------------------------
ax_vel = 1.0
ay_vel = 1.0

# Grid size
nx = 120
ny = 120

dx = 1.0 / (nx - 1)
dy = 1.0 / (ny - 1)

dt = 0.01

# CFL numbers
lam_x = ax_vel * dt / dx
lam_y = ay_vel * dt / dy

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
# Initial condition (Top Hat)
# --------------------------------------------------
u_initial = da.createGlobalVec()
(xs, xe), (ys, ye) = da.getRanges()

with da.getVecArray(u_initial) as arr:
    for j in range(ys, ye):
        for i in range(xs, xe):
            x = i * dx
            y = j * dy
            # Top Hat Logic: Square pulse centered at (0.25, 0.25)
            # Width = 0.2 (from 0.15 to 0.35)
            if abs(x) < 0.5 and abs(y) < 0.5:
                arr[j, i] = 1.0   # High
            else:
                arr[j, i] = 0.0   # Low

# DEBUG: Print mass to prove initial condition exists
total_mass = u_initial.sum()
print(f"Initial Total Mass: {total_mass:.4f} (If 0.0, something is wrong)")

# --------------------------------------------------
# Matrix A (Implicit Upwind)
# --------------------------------------------------
A = da.createMatrix()
row = PETSc.Mat.Stencil()
col = PETSc.Mat.Stencil()

diag_val = 1.0 + lam_x + lam_y

for j in range(ys, ye):
    for i in range(xs, xe):
        row.index = (i, j)
        col.index = (i, j)
        A.setValueStencil(row, col, diag_val)

        if i - 1 >= 0:
            col.index = (i - 1, j)
            A.setValueStencil(row, col, -lam_x)
            
        if j - 1 >= 0:
            col.index = (i, j - 1)
            A.setValueStencil(row, col, -lam_y)

A.assemblyBegin()
A.assemblyEnd()

# --------------------------------------------------
# Linear solver
# --------------------------------------------------
ksp = PETSc.KSP().create(comm=A.getComm())
ksp.setOperators(A)
ksp.setType(PETSc.KSP.Type.GMRES)
pc = ksp.getPC()
pc.setType(PETSc.PC.Type.JACOBI)
pc.setFromOptions()
ksp.setFromOptions()

# --------------------------------------------------
# Time Stepping
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
    u.copy(b)            # RHS = u^n
    ksp.solve(b, u_new)  # Solve for u^{n+1}
    
    # --- FIXED BUG HERE ---
    # We copy the NEW solution (u_new) INTO the OLD variable (u)
    u_new.copy(u)        
    # ----------------------

    current_time = step * dt
    times.append(current_time)
    
    # Store Data
    sol_array = u.getArray().reshape(ny, nx).copy()
    solution_history.append(sol_array)
    slice_history.append(sol_array[mid_y, :])
    
    # Velocity Tracking
    mass = np.sum(sol_array)
    if mass > 1e-10:
        Y_grid, X_grid = np.indices((ny, nx))
        x_cm = np.sum(X_grid * sol_array) / mass * dx
        y_cm = np.sum(Y_grid * sol_array) / mass * dy
        curr_cent = np.array([x_cm, y_cm])
        
        if prev_centroid is not None:
            velocities.append(np.linalg.norm(curr_cent - prev_centroid) / dt)
        else:
            velocities.append(np.sqrt(ax_vel**2 + ay_vel**2))
        prev_centroid = curr_cent
    else:
        velocities.append(0.0)

print(f"Iterations (Last Step) = {ksp.getIterationNumber()}")

# ==============================================================================
# PLOTTING
# ==============================================================================

# 1. Animation
print("Generating Animation...")
fig_anim, ax_anim = plt.subplots(figsize=(6, 5))
im = ax_anim.imshow(solution_history[0], origin='lower', extent=[0, 1, 0, 1], 
               cmap='viridis', vmin=0, vmax=1.0)
fig_anim.colorbar(im, ax=ax_anim, label='u')
ax_anim.set_title("2D Advection (Top Hat)")

def update(frame):
    im.set_data(solution_history[frame])
    ax_anim.set_title(f"Time = {frame * dt:.2f}")
    return im,

ani = FuncAnimation(fig_anim, update, frames=nt, interval=50, blit=False)
ani.save("2d-advection-upwind-tophat-heatmap.gif", dpi=100)
plt.close(fig_anim)
print("-> Saved .gif")

# 2. Space-Time Surface
print("Generating Space-Time Surface...")
fig_st = plt.figure(figsize=(10, 7))
ax_st = fig_st.add_subplot(111, projection='3d')

x_vals = np.linspace(0, 1, nx)
t_vals = np.linspace(0, nt*dt, nt)
X_mesh, T_mesh = np.meshgrid(x_vals, t_vals)
Z_mesh = np.array(slice_history) 

surf = ax_st.plot_surface(X_mesh, T_mesh, Z_mesh, cmap=cm.viridis, linewidth=0, antialiased=False)
ax_st.set_xlabel('Space (x)')
ax_st.set_ylabel('Time (t)')
ax_st.set_zlabel('u')
ax_st.set_title(f'Space-Time Evolution (Slice at y={mid_y*dy:.2f})')
fig_st.colorbar(surf, shrink=0.5, aspect=5, label='u')
ax_st.view_init(elev=30, azim=-60)

plt.savefig("2d-advection-upwind-tophat-surface.png", dpi=150)
plt.close(fig_st)
print("-> Saved surface plot")

# 3. 1D Snapshots
print("Generating 1D Snapshots...")
fig_slice, ax_slice = plt.subplots(figsize=(8, 5))
indices = np.linspace(0, nt-1, 5, dtype=int)
for idx in indices:
    ax_slice.plot(x_vals, slice_history[idx], lw=2, label=f't={idx*dt:.2f}')
ax_slice.set_title("1D Profiles (Top Hat)")
ax_slice.legend()
plt.savefig("2d-advection-upwind-tophat-1d-slicing.png", dpi=150)
plt.close(fig_slice)
print("-> Saved 1D slices")

# 4. Speed
print("Generating Speed Plot...")
fig_speed, ax_speed = plt.subplots(figsize=(8, 5))
ax_speed.plot(times, velocities, 'r-', marker='o', markersize=3)
ax_speed.set_title("Wave Speed vs Time")
plt.savefig("2d-advection-upwind-tophat-ut.png", dpi=150)
plt.close(fig_speed)
print("Done.")