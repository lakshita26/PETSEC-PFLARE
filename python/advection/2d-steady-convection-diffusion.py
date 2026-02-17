import numpy as np
import matplotlib.pyplot as plt
import sys

try:
    import petsc4py
    petsc4py.init(sys.argv)
    from petsc4py import PETSc
except ModuleNotFoundError:
    print("petsc4py not found")
    sys.exit()

# --------------------------------------------------
# Parameters
# --------------------------------------------------
ax = 2.0          # Advection velocity in x
ay = 1.0          # Advection velocity in y
nu = 0.05         # Diffusion

nx, ny = 64, 64
dx = 1.0 / (nx - 1)
dy = 1.0 / (ny - 1)

phi_left  = 1.0
phi_right = 0.0

# --------------------------------------------------
# DMDA grid (2D)
# --------------------------------------------------
da = PETSc.DMDA().create(
    sizes=[nx, ny],
    dof=1,
    stencil_width=1,
    boundary_type=PETSc.DM.BoundaryType.NONE
)

# --------------------------------------------------
# Matrix Assembly
# --------------------------------------------------
A = da.createMatrix()
(xs, xe), (ys, ye) = da.getRanges()

row = PETSc.Mat.Stencil()
col = PETSc.Mat.Stencil()

for j in range(ys, ye):
    for i in range(xs, xe):
        row.index = (i, j)

        # Dirichlet boundaries
        if i == 0:
            A.setValueStencil(row, row, 1.0)

        elif i == nx - 1 or j == 0 or j == ny - 1:
            A.setValueStencil(row, row, 1.0)

        else:
            # Diagonal
            col.index = (i, j)
            val_diag = (
                ax / dx + ay / dy
                + 2.0 * nu * (1.0 / dx**2 + 1.0 / dy**2)
            )
            A.setValueStencil(row, col, val_diag)

            # West (upwind in x)
            col.index = (i - 1, j)
            A.setValueStencil(
                row, col,
                -(ax / dx + nu / dx**2)
            )

            # East
            col.index = (i + 1, j)
            A.setValueStencil(
                row, col,
                -nu / dx**2
            )

            # South (upwind in y)
            col.index = (i, j - 1)
            A.setValueStencil(
                row, col,
                -(ay / dy + nu / dy**2)
            )

            # North
            col.index = (i, j + 1)
            A.setValueStencil(
                row, col,
                -nu / dy**2
            )

A.assemblyBegin()
A.assemblyEnd()

subset = 10 
# Fetch the values for the top-left corner
dense_subset = A.getValues(range(subset), range(subset))

print(f"\n--- Top-Left {subset}x{subset} Block of Matrix A ---")
print(np.array2string(dense_subset, precision=2, suppress_small=True, separator=', '))# Pick an interior starting point (i=10, j=10)
# Global index = j * nx + i
start_i, start_j = 10, 10
start_index = start_j * nx + start_i

# We want to see a small block of the matrix around this interior point
# Let's look at 5 consecutive rows to see how the stencil shifts
subset_size = 5
rows = range(start_index, start_index + subset_size)
cols = range(start_index - 1, start_index + subset_size + 1) # Extra column for West neighbor

# Fetch and print
dense_interior = A.getValues(rows, cols)

print(f"\n--- Interior {subset_size}x{len(cols)} Block (Starting at i={start_i}, j={start_j}) ---")
print("Note: This shows the repeating 5-point stencil values.")
print(np.array2string(dense_interior, precision=2, suppress_small=True, separator=', '))

# --------------------------------------------------
# RHS Assembly
# --------------------------------------------------
b = da.createGlobalVec()
b.set(0.0)

(xs, xe), (ys, ye) = da.getRanges()
with da.getVecArray(b) as b_arr:
    for j in range(ys, ye):
        if xs <= 0 < xe:
            b_arr[0, j] = phi_left
        if xs <= nx - 1 < xe:
            b_arr[nx - 1, j] = phi_right

# --------------------------------------------------
# Solver
# --------------------------------------------------
u_sol = da.createGlobalVec()

ksp = PETSc.KSP().create(comm=A.getComm())
ksp.setOperators(A)
ksp.setType(PETSc.KSP.Type.GMRES)
ksp.getPC().setType(PETSc.PC.Type.GAMG)
ksp.setFromOptions()

ksp.solve(b, u_sol)

print(f"Convergence Reason: {ksp.getConvergedReason()}")
print(f"Iterations = {ksp.getIterationNumber()}")

# --------------------------------------------------
# Visualization
# --------------------------------------------------
solution = u_sol.getArray().reshape(nx, ny).T

# ----- 2D Plot (unchanged) -----
plt.figure(figsize=(6, 5))
plt.imshow(solution, origin='lower', extent=[0, 1, 0, 1], cmap='magma')
plt.colorbar(label='u')
plt.title("2D Steady Convection–Diffusion")
plt.xlabel("x")
plt.ylabel("y")
plt.savefig("2d-steady-convection-diffusion.png")

# --------------------------------------------------
# 3D Surface Plot
# --------------------------------------------------
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 7))
ax_3d = fig.add_subplot(111, projection='3d')

# Create the meshgrid for the x and y axes
x_coords = np.linspace(0.0, 1.0, nx)
y_coords = np.linspace(0.0, 1.0, ny)
X, Y = np.meshgrid(x_coords, y_coords)

# Plot the surface
surf = ax_3d.plot_surface(X, Y, solution, cmap='magma', edgecolor='none', alpha=0.9)

# Add labels and formatting
ax_3d.set_title("3D Surface Plot of Steady Convection–Diffusion")
ax_3d.set_xlabel("x")
ax_3d.set_ylabel("y")
ax_3d.set_zlabel("u")
fig.colorbar(surf, ax=ax_3d, shrink=0.5, aspect=10, label='u')

plt.tight_layout()
plt.savefig("2d-steady-surface-plot.png")

# ----- 1D Line Plot (centerline slice) -----
mid_y = ny // 2
slice_x = solution[mid_y, :]

x_axis = np.linspace(0.0, 1.0, nx)

plt.figure(figsize=(8, 5))
plt.plot(x_axis, slice_x, 'o-', markersize=3, label='y = 0.5 slice')
plt.title("1D Slice from 2D Steady Convection–Diffusion")
plt.xlabel("x")
plt.ylabel("u")
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig("2d-steady-convection-diffusion-centerline_slice.png")