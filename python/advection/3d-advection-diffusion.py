import sys
import numpy as np
import matplotlib.pyplot as plt
from petsc4py import PETSc
from mpl_toolkits.mplot3d import Axes3D

# --------------------------------------------------
# 1. Parameters
# --------------------------------------------------
nx, ny, nz = 50, 50, 50
dx, dy, dz = 1.0/(nx-1), 1.0/(ny-1), 1.0/(nz-1)

# Physics
nu = 0.0001
ax_vel, ay_vel, az_vel = 1.0, 1.0, 1.0
dt = 0.005
nt = 50

# Initial Condition Config
box_center = 0.25
box_width = 0.2
box_min = box_center - box_width/2
box_max = box_center + box_width/2

# --------------------------------------------------
# 2. Grid Setup
# --------------------------------------------------
da = PETSc.DMDA().create(sizes=[nx, ny, nz], dof=1, stencil_width=1, stencil_type=PETSc.DMDA.StencilType.STAR)
u = da.createGlobalVec()
u_initial = da.createGlobalVec()
u_new = da.createGlobalVec()
rhs = da.createGlobalVec()

# --------------------------------------------------
# 3. Vectorized Initial Condition (NO LOOPS)
# --------------------------------------------------
(xs, xe), (ys, ye), (zs, ze) = da.getRanges()

with da.getVecArray(u) as arr:
    # 1. Create local coordinate grids (Vectorized)
    # indexing='ij' ensures (z, y, x) ordering to match PETSc
    Z, Y, X = np.meshgrid(
        np.arange(zs, ze) * dz,
        np.arange(ys, ye) * dy,
        np.arange(xs, xe) * dx,
        indexing='ij'
    )
    
    # 2. Define the Mask (The Shape Logic)
    # This replaces the manual "if" statement with a mathematical condition
    mask = (X >= box_min) & (X <= box_max) & \
           (Y >= box_min) & (Y <= box_max) & \
           (Z >= box_min) & (Z <= box_max)
    
    # 3. Apply Values using the Mask
    arr[...] = 0.0      # Initialize background
    arr[mask] = 1.0     # Set Top-Hat

# Save t=0 state
u.copy(u_initial)

# --------------------------------------------------
# 4. Matrix Assembly
# --------------------------------------------------
A = da.createMatrix()
row = PETSc.Mat.Stencil()
col = PETSc.Mat.Stencil()

adv_x, adv_y, adv_z = ax_vel * dt / dx, ay_vel * dt / dy, az_vel * dt / dz
diff_x, diff_y, diff_z = nu * dt / dx**2, nu * dt / dy**2, nu * dt / dz**2

for k in range(zs, ze):
    for j in range(ys, ye):
        for i in range(xs, xe):
            row.index = (i, j, k)
            
            if (i==0 or i==nx-1 or j==0 or j==ny-1 or k==0 or k==nz-1):
                col.index = (i, j, k); A.setValueStencil(row, col, 1.0)
            else:
                val_center = 1.0 + (adv_x + adv_y + adv_z) + 2.0*(diff_x + diff_y + diff_z)
                col.index = (i, j, k); A.setValueStencil(row, col, val_center)
                col.index = (i-1, j, k); A.setValueStencil(row, col, -(adv_x + diff_x))
                col.index = (i+1, j, k); A.setValueStencil(row, col, -diff_x)
                col.index = (i, j-1, k); A.setValueStencil(row, col, -(adv_y + diff_y))
                col.index = (i, j+1, k); A.setValueStencil(row, col, -diff_y)
                col.index = (i, j, k-1); A.setValueStencil(row, col, -(adv_z + diff_z))
                col.index = (i, j, k+1); A.setValueStencil(row, col, -diff_z)

A.assemblyBegin()
A.assemblyEnd()

ksp = PETSc.KSP().create()
ksp.setOperators(A)
ksp.setType(PETSc.KSP.Type.GMRES)
ksp.getPC().setType(PETSc.PC.Type.GAMG)
ksp.setFromOptions()

# --------------------------------------------------
# 5. Time Stepping
# --------------------------------------------------
print(f"Solving {nt} steps...")
for step in range(nt):
    u.copy(rhs)
    ksp.solve(rhs, u_new)
    u_new.copy(u)

# --------------------------------------------------
# 6. Verification Plot
# --------------------------------------------------
sol_init = u_initial.getArray().reshape(nz, ny, nx)
sol_final = u.getArray().reshape(nz, ny, nx)

z_slice_idx = int(box_center * (nz-1)) 
y_slice_idx = int(box_center * (ny-1))
line_initial = sol_init[z_slice_idx, y_slice_idx, :]

z_final_idx, y_final_idx = np.unravel_index(sol_final.argmax(), sol_final.shape)[0:2]
line_final_tracked = sol_final[z_final_idx, y_final_idx, :]

x_axis = np.linspace(0, 1, nx)

plt.figure(figsize=(10, 6))
plt.plot(x_axis, line_initial, 'k--', linewidth=2, label='Initial (t=0)')
plt.plot(x_axis, line_final_tracked, 'b-', linewidth=2, label=f'Final (t={nt*dt})')
plt.title(f"Advection of Top-Hat Profile")
plt.xlabel("X Coordinate")
plt.ylabel("Concentration u")
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)

filename = "3d_advection_diffusion_line-tophat.png"
plt.savefig(filename, dpi=150)
print(f"Plot saved to {filename}")