import os
import numpy as np
import pandas as pd
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from petsc4py import PETSc
import pflare

# --- 1. Setup ---
is_gpu = torch.cuda.is_available()
device = torch.device("cuda" if is_gpu else "cpu")

dt = 0.05
dx = 1.0
dy = 1.0
Re = 0.0
# Re = 0.5
ub = 1.0
nx = 256
ny = 256
ntime = 100

N = nx * ny

# --- Meshgrid for UNIT_TEST ---
Y, X = np.meshgrid(np.arange(ny)*dy, np.arange(nx)*dx, indexing="ij")

# -----------------------
# Index mapping
# -----------------------
def idx(i, j):
    return i * nx + j

# -----------------------
# UNIT TEST (same logic as code 1)
# -----------------------
def UNIT_TEST(c_vec):
    c = c_vec.reshape(ny, nx)
    mass = np.sum(c)*dx*dy
    if mass < 1e-5:
        return 0.0, 0.0, 0.0, 0.0
    x_com = np.sum(c*X)*dx*dy/mass
    y_com = np.sum(c*Y)*dx*dy/mass
    var = np.sum(c*((X-x_com)**2+(Y-y_com)**2))*dx*dy/mass
    return mass, x_com, y_com, var

# -----------------------
# Build Explicit Sparse Matrix
# -----------------------
A = PETSc.Mat().createAIJ([N, N], nnz=5)
A.setUp()

lap_center = -4.0 / (dx*dx)
lap_side   =  1.0 / (dx*dx)
adv_x = ub / (2.0*dx)
adv_y = ub / (2.0*dy)

for i in range(ny):
    for j in range(nx):

        row = idx(i, j)
        diag = 1.0 - dt*(Re*lap_center)
        A.setValue(row, row, diag)

        if j > 0:
            A.setValue(row, idx(i,j-1), -dt*(Re*lap_side + adv_x))
        if j < nx-1:
            A.setValue(row, idx(i,j+1), -dt*(Re*lap_side - adv_x))
        if i > 0:
            A.setValue(row, idx(i-1,j), -dt*(Re*lap_side + adv_y))
        if i < ny-1:
            A.setValue(row, idx(i+1,j), -dt*(Re*lap_side - adv_y))

A.assemblyBegin()
A.assemblyEnd()

# -----------------------
# Initial Condition
# -----------------------
u = PETSc.Vec().createSeq(N)
rhs = PETSc.Vec().createSeq(N)

initial = np.zeros(N)
for i in range(ny//2 - 25, ny//2 + 26):
    for j in range(nx//2 - 25, nx//2 + 26):
        initial[idx(i,j)] = 1.0

u.setArray(initial.copy())

# -----------------------
# KSP + AIR Preconditioner
# -----------------------
ksp = PETSc.KSP().create()
ksp.setOperators(A)
ksp.setType("gmres")

pc = ksp.getPC()
pc.setType("air")   # keep your AIR

ksp.setTolerances(rtol=1e-6)
ksp.setFromOptions()

# -----------------------
# Printing Header (same as code 1)
# -----------------------
print(f"{'Step':<5} | {'Residual':<10} | {'Step Time':<10} | {'Mass':<10} | {'Max_Val':<10}")
print("-" * 60)

mass, _, _, _ = UNIT_TEST(initial)
print(f"{0:<5} | {'---':<10} | {'---':<10} | {mass:<10.2f} | {initial.max():<10.2f}")

# -----------------------
# Time Stepping
# -----------------------
total_start = time.time()
frames = []

for t in range(1, ntime+1):

    u.copy(rhs)

    step_start = time.time()
    ksp.solve(rhs, u)
    step_elapsed = time.time() - step_start

    current = u.getArray().copy()
    frames.append(current.reshape(ny, nx))

    mass, _, _, _ = UNIT_TEST(current)

    print(f"{t:<5} | {ksp.getResidualNorm():<10.2e} | {step_elapsed:<10.4f} | {mass:<10.2f} | {current.max():<10.2f}")

total_elapsed = time.time() - total_start

print("-" * 60)
print(f"Total Elapsed Time: {total_elapsed:.4f} seconds")
print("-" * 60)

print("Simulation Complete.")

# Animation
import matplotlib.animation as animation

fig, ax = plt.subplots(figsize=(6,5))
img = ax.imshow(frames[0], cmap='jet', origin='lower')
ax.set_title("Implicit Advection-Diffusion")

def update(frame):
    img.set_array(frame)
    return [img]

ani = animation.FuncAnimation(
    fig,
    update,
    frames=frames,
    interval=50,   # milliseconds between frames
    blit=True
)
ani.save("solution_animation.gif", writer="pillow", fps=20)
plt.close()


# Visualization
plt.figure(figsize=(6, 5))

# reshape final solution
final_field = current.reshape(ny, nx)

img = plt.imshow(final_field, cmap='jet', origin='lower')
plt.colorbar(img, label='Concentration')

plt.title(f"Implicit Solution (t={ntime*dt:.2f})")

if Re > 0.0:
    plt.savefig('solution_Re_pflare.png')
else:
    plt.savefig('solution_pure_pflare.png')

plt.close()

# ---- BUILD SMALL EXPLICIT A MATRIX USING PETSc ----
nx_small = 3
ny_small = 3
N_small = nx_small * ny_small

def idx_small(i, j):
    return i * nx_small + j

A_small = np.zeros((N_small, N_small))

for i in range(ny_small):
    for j in range(nx_small):

        row = idx_small(i, j)
        diag = 1.0 - dt*(Re*lap_center)
        A_small[row, row] = diag

        if j > 0:
            A_small[row, idx_small(i,j-1)] = -dt*(Re*lap_side + adv_x)
        if j < nx_small-1:
            A_small[row, idx_small(i,j+1)] = -dt*(Re*lap_side - adv_x)
        if i > 0:
            A_small[row, idx_small(i-1,j)] = -dt*(Re*lap_side + adv_y)
        if i < ny_small-1:
            A_small[row, idx_small(i+1,j)] = -dt*(Re*lap_side - adv_y)

np.set_printoptions(precision=4, suppress=True, linewidth=150)

print("\nExplicit Small A matrix:\n")
print(A_small)