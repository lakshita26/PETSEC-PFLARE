import sys
import numpy as np
import matplotlib.pyplot as plt
from petsc4py import PETSc

'''
=====================================================================
2D Linear Viscous Advection Equation (Advection-Diffusion)
Equation: u_t + a_x*u_x + a_y*u_y = nu * (u_xx + u_yy)

Implicit Scheme (Backward Euler)
- Time Integration: Backward Euler
- Advection: First-order Upwind
- Diffusion: Second-order Central Difference

Solver: GMRES
Preconditioner: GAMG
=====================================================================
'''

# --------------------------------------------------
# Parameters
# --------------------------------------------------
nx, ny = 64, 64
dx, dy = 1.0/(nx-1), 1.0/(ny-1)
ax_vel, ay_vel = 1.0, 1.0
nu = 0.005
dt = 0.01
nt = 50

# --------------------------------------------------
# DMDA grid setup
# --------------------------------------------------
da = PETSc.DMDA().create(sizes=[nx, ny], dof=1, stencil_width=1)
u = da.createGlobalVec()
u_old = da.createGlobalVec()
u_new = da.createGlobalVec()
rhs = da.createGlobalVec()

# Initial Condition (Gaussian Pulse)
(xs, xe), (ys, ye) = da.getRanges()
with da.getVecArray(u) as arr:
    for j in range(ys, ye):
        for i in range(xs, xe):
            r2 = (i*dx - 0.25)**2 + (j*dy - 0.25)**2
            arr[i, j] = np.exp(-r2 / (2 * 0.08**2))

# --------------------------------------------------
# Matrix A Assembly
# --------------------------------------------------
A = da.createMatrix()
row = PETSc.Mat.Stencil()
col = PETSc.Mat.Stencil()

for j in range(ys, ye):
    for i in range(xs, xe):
        row.index = (i, j)
        if i == 0 or i == nx-1 or j == 0 or j == ny-1:
            col.index = (i, j)
            A.setValueStencil(row, col, 1.0)
        else:
            adv_x, adv_y = ax_vel * dt / dx, ay_vel * dt / dy
            diff_x, diff_y = nu * dt / dx**2, nu * dt / dy**2

            # Center
            col.index = (i, j)
            A.setValueStencil(row, col, 1.0 + adv_x + adv_y + 2.0*diff_x + 2.0*diff_y)
            # West
            col.index = (i-1, j); A.setValueStencil(row, col, -(adv_x + diff_x))
            # South
            col.index = (i, j-1); A.setValueStencil(row, col, -(adv_y + diff_y))
            # East
            col.index = (i+1, j); A.setValueStencil(row, col, -diff_x)
            # North
            col.index = (i, j+1); A.setValueStencil(row, col, -diff_y)

A.assemblyBegin()
A.assemblyEnd()

# Solver
ksp = PETSc.KSP().create()
ksp.setOperators(A)
ksp.setType(PETSc.KSP.Type.GMRES)
ksp.getPC().setType(PETSc.PC.Type.GAMG)
ksp.setFromOptions()

# --------------------------------------------------
# Time Stepping
# --------------------------------------------------
for step in range(nt):
    u.copy(u_old) # Keep track of u_n
    u.copy(rhs)   # Set RHS for A * u_{n+1} = u_n
    ksp.solve(rhs, u_new)
    u_new.copy(u)

# --------------------------------------------------
# Convergence & Residual Diagnostics
# --------------------------------------------------
print(f"Iterations = {ksp.getIterationNumber()}")
print(f"Final Linear Residual Norm = {ksp.getResidualNorm():.3e}")

# Compute PDE Residual: R = (u_new - u_old)/dt + a.grad(u_new) - nu.laplacian(u_new)
res_vec = da.createGlobalVec()
with da.getVecArray(u) as u_arr, da.getVecArray(u_old) as u_o_arr, da.getVecArray(res_vec) as r_arr:
    for j in range(ys, ye):
        for i in range(xs, xe):
            if 0 < i < nx-1 and 0 < j < ny-1:
                ut = (u_arr[i, j] - u_o_arr[i, j]) / dt
                ux = (u_arr[i, j] - u_arr[i-1, j]) / dx
                uy = (u_arr[i, j] - u_arr[i, j-1]) / dy
                lap = (u_arr[i+1, j] - 2*u_arr[i, j] + u_arr[i-1, j])/dx**2 + \
                      (u_arr[i, j+1] - 2*u_arr[i, j] + u_arr[i, j-1])/dy**2
                r_arr[i, j] = ut + ax_vel*ux + ay_vel*uy - nu*lap
            else:
                r_arr[i, j] = 0.0 # Boundary residuals ignored for this check

print(f"[PETSc] PDE Residual Norm = {res_vec.norm():.3e}")

# --------------------------------------------------
# Final Plot
# --------------------------------------------------
plt.figure(figsize=(6, 5))
data = u.getArray().reshape(xe-xs, ye-ys).T
plt.imshow(data, origin='lower', extent=[0, 1, 0, 1], cmap='magma')
plt.colorbar(label='u')
plt.title(f"2D Advection-Diffusion | Step {nt}")
plt.xlabel("x")
plt.ylabel("y")

filename = "2d_advection_diffusion.png"
plt.savefig(filename, dpi=150)
print(f"Plot saved to {filename}")


# --------------------------------------------------
# Line Graph
# --------------------------------------------------
# Reshape to 2D arrays (Transpose needed because PETSc is (y,x) vs Matplotlib (x,y))
sol_final = u.getArray().reshape(xe-xs, ye-ys).T
sol_initial = u_old.getArray().reshape(xe-xs, ye-ys).T

# Take a slice through the center Y-axis
mid_y_index = (ye - ys) // 2
slice_initial = sol_initial[mid_y_index, :]
slice_final   = sol_final[mid_y_index, :]

x_axis = np.linspace(0, 1, nx)

plt.figure(figsize=(8, 5))

# Plot Initial Condition
plt.plot(x_axis, slice_initial, linestyle='--', color='grey', label='Initial (t=0)')

# Plot Final Solution
plt.plot(x_axis, slice_final, color='blue', linewidth=2, label=f'Final (t={nt*dt})')

plt.title(f"Cross-Section at y=0.5\n(Advection velocity = {ax_vel})")
plt.xlabel("x coordinate")
plt.ylabel("u (Concentration)")
plt.legend()
plt.grid(True, alpha=0.3)

# Save
filename = "2d-advection-diffusion-line.png"
plt.savefig(filename, dpi=150)
print(f"Line graph saved to {filename}")