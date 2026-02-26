import numpy as np
import matplotlib.pyplot as plt
import sys
import time

try:
    import petsc4py
    petsc4py.init(sys.argv)
    from petsc4py import PETSc
except ModuleNotFoundError:
    print("petsc4py not found")
    sys.exit()

'''
2D Steady Pure Advection Operator

2u(i,j) - u(i-1,j) - u(i,j-1)
'''

# ==================================================
# Start total timer
# ==================================================

t_total_start = time.time()

# --------------------------------------------------
# Parameters
# --------------------------------------------------

nx = 50
ny = 50
length_x = 1.0
length_y = 1.0

dx = length_x / (nx - 1)
dy = length_y / (ny - 1)

# --------------------------------------------------
# DMDA grid
# --------------------------------------------------

da = PETSc.DMDA().create(
    sizes=[nx, ny],
    dof=1,
    stencil_width=1,
    boundary_type=(
        PETSc.DM.BoundaryType.NONE,
        PETSc.DM.BoundaryType.NONE,
    )
)

da.setUniformCoordinates(0.0, length_x, 0.0, length_y)

# --------------------------------------------------
# Matrix & vectors (GPU)
# --------------------------------------------------

A = da.createMatrix()
A.setType(PETSc.Mat.Type.AIJCUSPARSE)

b = da.createGlobalVec()
u = da.createGlobalVec()

b.setType(PETSc.Vec.Type.SEQCUDA)
u.setType(PETSc.Vec.Type.SEQCUDA)

(xs, xe), (ys, ye) = da.getRanges()

row = PETSc.Mat.Stencil()
col = PETSc.Mat.Stencil()

# ==================================================
# Assembly timing
# ==================================================

t_assembly_start = time.time()

with da.getVecArray(b) as b_arr:
    for j in range(ys, ye):
        for i in range(xs, xe):

            row.index = (i, j)
            A.setValueStencil(row, row, 2.0)

            if i > 0:
                col.index = (i - 1, j)
                A.setValueStencil(row, col, -1.0)

            if j > 0:
                col.index = (i, j - 1)
                A.setValueStencil(row, col, -1.0)

            b_arr[i, j] = 1.0

A.assemblyBegin()
A.assemblyEnd()

b.assemblyBegin()
b.assemblyEnd()

# Force sync for accurate timing
b.norm()

t_assembly_end = time.time()

# --------------------------------------------------
# Solver
# --------------------------------------------------

ksp = PETSc.KSP().create(comm=da.getComm())
ksp.setOperators(A)
ksp.setType(PETSc.KSP.Type.GMRES)
ksp.setTolerances(rtol=0.0, atol=1e-5)

pc = ksp.getPC()
pc.setType(PETSc.PC.Type.JACOBI)

# ==================================================
# Solver timing
# ==================================================

t_solve_start = time.time()

ksp.solve(b, u)

# Force GPU sync
u.norm()

t_solve_end = time.time()

# --------------------------------------------------
# Solver statistics
# --------------------------------------------------

print("\nSolver statistics")
print("Iterations:", ksp.getIterationNumber())
print("Residual:", ksp.getResidualNorm())
print("Converged:", ksp.getConvergedReason())

# --------------------------------------------------
# Copy solution to CPU
# --------------------------------------------------

u_cpu = u.copy()
u_cpu.setType(PETSc.Vec.Type.SEQ)

u_arr = u_cpu.getArray().reshape((ny, nx))

# --------------------------------------------------
# Plotting
# --------------------------------------------------

t_plot_start = time.time()

plt.figure(figsize=(6, 5))
plt.imshow(
    u_arr,
    origin="lower",
    extent=[0.0, length_x, 0.0, length_y],
    aspect="auto"
)
plt.colorbar(label="u")
plt.xlabel("x")
plt.ylabel("y")
plt.title("2D Steady Advection Solution")
plt.tight_layout()
plt.savefig("2d-steady-advection.png", dpi=150)
plt.close()

print("Saved: 2d-steady-advection.png")

mid_j = ny // 2
centerline = u_arr[mid_j, :]

plt.figure(figsize=(6, 4))
plt.plot(np.linspace(0.0, length_x, nx), centerline, lw=2)
plt.xlabel("x")
plt.ylabel("u")
plt.title(f"Centerline y = {mid_j * dy:.2f}")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("2d-steady-advection-centerline.png", dpi=150)
plt.close()

print("Saved: 2d-steady-advection-centerline.png")

t_plot_end = time.time()

# ==================================================
# Final timing report
# ==================================================

t_total_end = time.time()

print("\nTiming report")
print(f"Assembly time : {t_assembly_end - t_assembly_start:.3f} s")
print(f"Solve time    : {t_solve_end - t_solve_start:.3f} s")
print(f"Plot time     : {t_plot_end - t_plot_start:.3f} s")
print(f"Total runtime : {t_total_end - t_total_start:.3f} s")
