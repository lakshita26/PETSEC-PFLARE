import sys
import numpy as np
import matplotlib.pyplot as plt
from petsc4py import PETSc
from mpl_toolkits.mplot3d import Axes3D

"""
=====================================================================
3D Linear Viscous Advection Equation (Top-Hat Profile)
Equation: u_t + a·∇u = ν ∇²u

Scheme:
- Backward Euler (implicit)
- First-order upwind (positive velocity)
- Second-order central diffusion
- Vectorized Initial Condition (No Loops)
=====================================================================
"""

# --------------------------------------------------
# 1. Parameters
# --------------------------------------------------
nx, ny, nz = 50, 50, 50
dx, dy, dz = 1.0/(nx-1), 1.0/(ny-1), 1.0/(nz-1)

ax_vel, ay_vel, az_vel = 1.0, 1.0, 1.0
nu = 0.0001        # Low diffusion to keep edges sharp
dt = 0.005
nt = 50

# Top-hat configuration
box_center = 0.25
box_width  = 0.20
val_inside = 1.0
val_outside = 0.0

box_min = box_center - box_width/2
box_max = box_center + box_width/2

# --------------------------------------------------
# 2. DMDA Grid Setup
# --------------------------------------------------
da = PETSc.DMDA().create(
    sizes=[nx, ny, nz],
    dof=1,
    stencil_width=1,
    stencil_type=PETSc.DMDA.StencilType.STAR
)

u         = da.createGlobalVec()
u_initial = da.createGlobalVec()
u_new     = da.createGlobalVec()
rhs       = da.createGlobalVec()

(xs, xe), (ys, ye), (zs, ze) = da.getRanges()

# --------------------------------------------------
# 3. Vectorized Initial Condition (No Loops)
# --------------------------------------------------
with da.getVecArray(u) as arr:
    # Create local coordinate grids
    # indexing='ij' ensures (z, y, x) ordering to match PETSc
    Z, Y, X = np.meshgrid(
        np.arange(zs, ze) * dz,
        np.arange(ys, ye) * dy,
        np.arange(xs, xe) * dx,
        indexing='ij'
    )
    
    # Define Mask (Vectorized Logic)
    mask = (X >= box_min) & (X <= box_max) & \
           (Y >= box_min) & (Y <= box_max) & \
           (Z >= box_min) & (Z <= box_max)
    
    # Apply Values
    arr[...] = val_outside
    arr[mask] = val_inside

# Save t=0 state
u.copy(u_initial)

# --------------------------------------------------
# 4. Matrix Assembly
# --------------------------------------------------
A = da.createMatrix()
row = PETSc.Mat.Stencil()
col = PETSc.Mat.Stencil()

adv_x, adv_y, adv_z = ax_vel*dt/dx, ay_vel*dt/dy, az_vel*dt/dz
diff_x, diff_y, diff_z = nu*dt/dx**2, nu*dt/dy**2, nu*dt/dz**2

for k in range(zs, ze):
    for j in range(ys, ye):
        for i in range(xs, xe):
            row.index = (i, j, k)

            # Dirichlet BC
            if (i == 0 or i == nx-1 or
                j == 0 or j == ny-1 or
                k == 0 or k == nz-1):
                col.index = (i, j, k)
                A.setValueStencil(row, col, 1.0)
            else:
                center = 1.0 + (adv_x + adv_y + adv_z) + \
                         2.0*(diff_x + diff_y + diff_z)

                col.index = (i, j, k);     A.setValueStencil(row, col, center)
                col.index = (i-1, j, k);   A.setValueStencil(row, col, -(adv_x + diff_x))
                col.index = (i+1, j, k);   A.setValueStencil(row, col, -diff_x)
                col.index = (i, j-1, k);   A.setValueStencil(row, col, -(adv_y + diff_y))
                col.index = (i, j+1, k);   A.setValueStencil(row, col, -diff_y)
                col.index = (i, j, k-1);   A.setValueStencil(row, col, -(adv_z + diff_z))
                col.index = (i, j, k+1);   A.setValueStencil(row, col, -diff_z)

A.assemblyBegin()
A.assemblyEnd()

# --------------------------------------------------
# 5. Solver Setup
# --------------------------------------------------
ksp = PETSc.KSP().create()
ksp.setOperators(A)
ksp.setType(PETSc.KSP.Type.GMRES)
ksp.getPC().setType(PETSc.PC.Type.GAMG)
ksp.setFromOptions()

# --------------------------------------------------
# 6. Time Integration
# --------------------------------------------------
print(f"Solving {nt} steps...")
for step in range(nt):
    u.copy(rhs)
    ksp.solve(rhs, u_new)
    u_new.copy(u)

# --------------------------------------------------
# 7. Visualization
# --------------------------------------------------
sol = u.getArray().reshape(nz, ny, nx)
sol0 = u_initial.getArray().reshape(nz, ny, nx)

# Indices for slicing through the center of the box
z_slice_idx = int(box_center * (nz-1))
y_slice_idx = int(box_center * (ny-1))

# --- Plot 1: Line Graph ---
line_initial = sol0[z_slice_idx, y_slice_idx, :]
# Track peak for final solution
z_final_idx, y_final_idx = np.unravel_index(sol.argmax(), sol.shape)[0:2]
line_final_tracked = sol[z_final_idx, y_final_idx, :]
x_axis = np.linspace(0, 1, nx)

plt.figure(figsize=(10, 6))
plt.plot(x_axis, line_initial, 'k--', linewidth=2, label='Initial (t=0)')
plt.plot(x_axis, line_final_tracked, 'b-', linewidth=2, label=f'Final (t={nt*dt})')
plt.title("Advection of Top-Hat Profile")
plt.xlabel("X Coordinate")
plt.ylabel("Concentration u")
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)

filename_line = "3d_advection_diffusion_line-tophat.png"
plt.savefig(filename_line, dpi=150)
print(f"Line graph saved to {filename_line}")

# --- Plot 2: Surface Plot ---
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(np.linspace(0,1,nx), np.linspace(0,1,ny))
Z_slice_data = sol0[z_slice_idx, :, :]

ax.plot_surface(X, Y, Z_slice_data, cmap='viridis', linewidth=0)
ax.set_title("Initial Condition t=0 (Z-Slice)")
ax.set_zlim(0, 1.2)

filename_surf = "3d_advection_diffusion_surface-tophat.png"
plt.savefig(filename_surf, dpi=150)
print(f"Surface plot saved to {filename_surf}")