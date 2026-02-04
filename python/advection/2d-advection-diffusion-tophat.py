import sys
import numpy as np
import matplotlib.pyplot as plt
from petsc4py import PETSc

'''
=====================================================================
2D Linear Viscous Advection Equation (Advection-Diffusion)
u_t + a_x u_x + a_y u_y = nu (u_xx + u_yy)

Implicit Backward Euler
Upwind advection
Central diffusion
=====================================================================
'''

# --------------------------------------------------
# Parameters
# --------------------------------------------------
nx, ny = 64, 64
dx, dy = 2.0/(nx-1), 2.0/(ny-1)   # DOMAIN [-1,1]
ax_vel, ay_vel = 1.0, 1.0
nu = 0.005
dt = 0.01
nt = 50

# --------------------------------------------------
# DMDA grid setup
# --------------------------------------------------
da = PETSc.DMDA().create(
    sizes=[nx, ny],
    dof=1,
    stencil_width=1
)

u     = da.createGlobalVec()
u_old = da.createGlobalVec()
u_new = da.createGlobalVec()
rhs   = da.createGlobalVec()

# --------------------------------------------------
# Initial Condition (Top-Hat) — CORRECT
# --------------------------------------------------
(xs, xe), (ys, ye) = da.getRanges()
arr = da.getVecArray(u)

for j in range(ys, ye):
    for i in range(xs, xe):
        x = -1.0 + i * dx
        y = -1.0 + j * dy

        if abs(x) < 0.5 and abs(y) < 0.5:
            arr[i, j] = 1.0
        else:
            arr[i, j] = 0.0

# Save TRUE initial condition
u.copy(u_old)

# --------------------------------------------------
# Matrix Assembly
# --------------------------------------------------
A = da.createMatrix()
row = PETSc.Mat.Stencil()
col = PETSc.Mat.Stencil()

for j in range(ys, ye):
    for i in range(xs, xe):
        row.index = (i, j)

        if i == 0 or j == 0 or i == nx-1 or j == ny-1:
            col.index = (i, j)
            A.setValueStencil(row, col, 1.0)
        else:
            adv_x = ax_vel * dt / dx
            adv_y = ay_vel * dt / dy
            diff_x = nu * dt / dx**2
            diff_y = nu * dt / dy**2

            col.index = (i, j)
            A.setValueStencil(
                row, col,
                1.0 + adv_x + adv_y + 2*diff_x + 2*diff_y
            )

            col.index = (i-1, j)
            A.setValueStencil(row, col, -(adv_x + diff_x))

            col.index = (i, j-1)
            A.setValueStencil(row, col, -(adv_y + diff_y))

            col.index = (i+1, j)
            A.setValueStencil(row, col, -diff_x)

            col.index = (i, j+1)
            A.setValueStencil(row, col, -diff_y)

A.assemblyBegin()
A.assemblyEnd()

# --------------------------------------------------
# Solver
# --------------------------------------------------
ksp = PETSc.KSP().create()
ksp.setOperators(A)
ksp.setType(PETSc.KSP.Type.GMRES)
ksp.getPC().setType(PETSc.PC.Type.GAMG)
ksp.setFromOptions()

# --------------------------------------------------
# Time Stepping
# --------------------------------------------------
for step in range(nt):
    u.copy(rhs)
    ksp.solve(rhs, u_new)
    u_new.copy(u)

# --------------------------------------------------
# Diagnostics
# --------------------------------------------------
print(f"Iterations = {ksp.getIterationNumber()}")
print(f"Final Linear Residual Norm = {ksp.getResidualNorm():.3e}")

# --------------------------------------------------
# PDE Residual Check
# --------------------------------------------------
res_vec = da.createGlobalVec()
u_arr   = da.getVecArray(u)
u0_arr  = da.getVecArray(u_old)
r_arr   = da.getVecArray(res_vec)

for j in range(ys, ye):
    for i in range(xs, xe):
        if 0 < i < nx-1 and 0 < j < ny-1:
            ut = (u_arr[i, j] - u0_arr[i, j]) / (nt * dt)
            ux = (u_arr[i, j] - u_arr[i-1, j]) / dx
            uy = (u_arr[i, j] - u_arr[i, j-1]) / dy
            lap = (
                (u_arr[i+1, j] - 2*u_arr[i, j] + u_arr[i-1, j]) / dx**2 +
                (u_arr[i, j+1] - 2*u_arr[i, j] + u_arr[i, j-1]) / dy**2
            )
            r_arr[i, j] = ut + ax_vel*ux + ay_vel*uy - nu*lap
        else:
            r_arr[i, j] = 0.0

print(f"[PETSc] PDE Residual Norm = {res_vec.norm():.3e}")

# --------------------------------------------------
# 2D Plot
# --------------------------------------------------
plt.figure(figsize=(6,5))
data = u.getArray().reshape(nx, ny).T
plt.imshow(data, origin='lower', extent=[-1,1,-1,1], cmap='magma')
plt.colorbar(label='u')
plt.title(f"2D Advection-Diffusion | Step {nt}")
plt.xlabel("x")
plt.ylabel("y")

filename = "2d_advection_diffusion-tophat.png"
plt.savefig(filename, dpi=150)
print(f"Plot saved to {filename}")

# --------------------------------------------------
# Line Plot
# --------------------------------------------------
sol_initial = u_old.getArray().reshape(nx, ny).T
sol_final   = u.getArray().reshape(nx, ny).T

mid_y = ny // 2

plt.figure(figsize=(8,5))
plt.plot(np.linspace(-1,1,nx), sol_initial[mid_y, :],
         '--', color='gray', label='Initial (t=0)')
plt.plot(np.linspace(-1,1,nx), sol_final[mid_y, :],
         'b', lw=2, label=f'Final (t={nt*dt})')

plt.title(f"Cross-Section at y=0\n(Advection velocity = {ax_vel})")
plt.xlabel("x coordinate")
plt.ylabel("u")
plt.legend()
plt.grid(alpha=0.3)

filename = "2d-advection-diffusion-line-tophat.png"
plt.savefig(filename, dpi=150)
print(f"Line graph saved to {filename}")
