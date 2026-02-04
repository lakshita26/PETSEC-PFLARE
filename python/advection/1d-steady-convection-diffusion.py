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
a = 2.0          # Advection velocity
nu = 0.05        # Viscosity (Diffusion)
n = 100          # Grid points
dx = 1.0 / (n - 1)

# Boundary Values (Dirichlet)
phi_left = 1.0
phi_right = 0.0

# --------------------------------------------------
# DMDA grid
# --------------------------------------------------
da = PETSc.DMDA().create(
    sizes=[n],
    dof=1,
    stencil_width=1,
    boundary_type=PETSc.DM.BoundaryType.NONE
)

# --------------------------------------------------
# Matrix Assembly (Steady State)
# --------------------------------------------------
# Equation: a*u_x - nu*u_xx = 0
# Discretization: 
# Advection (Upwind): a * (u_i - u_{i-1}) / dx
# Diffusion (Central): -nu * (u_{i+1} - 2*u_i + u_{i-1}) / dx^2
# --------------------------------------------------

A = da.createMatrix()
(xs, xe) = da.getRanges()[0]

row = PETSc.Mat.Stencil()
col = PETSc.Mat.Stencil()

for i in range(xs, xe):
    row.index = (i,)
    
    if i == 0:
        # Left Boundary Condition (u = phi_left)
        A.setValueStencil(row, row, 1.0)
    elif i == n - 1:
        # Right Boundary Condition (u = phi_right)
        A.setValueStencil(row, row, 1.0)
    else:
        # Interior points: (a/dx + 2*nu/dx^2)u_i + (-a/dx - nu/dx^2)u_{i-1} + (-nu/dx^2)u_{i+1} = 0
        
        # Diagonal (u_i)
        col.index = (i,)
        val_diag = (a / dx) + (2.0 * nu / dx**2)
        A.setValueStencil(row, col, val_diag)

        # Left neighbor (u_{i-1})
        col.index = (i - 1,)
        val_left = (-a / dx) - (nu / dx**2)
        A.setValueStencil(row, col, val_left)

        # Right neighbor (u_{i+1})
        col.index = (i + 1,)
        val_right = -nu / dx**2
        A.setValueStencil(row, col, val_right)

A.assemblyBegin()
A.assemblyEnd()

# --------------------------------------------------
# Right-Hand Side (b) Assembly
# --------------------------------------------------
b = da.createGlobalVec()
with da.getVecArray(b) as b_arr:
    # Most interior points are 0 for steady state
    if xs <= 0 < xe:
        b_arr[0] = phi_left
    if xs <= n-1 < xe:
        b_arr[n-1] = phi_right

# --------------------------------------------------
# Solver Setup (GMRES + AMG)
# --------------------------------------------------
u_sol = da.createGlobalVec()

ksp = PETSc.KSP().create(comm=A.getComm())
ksp.setOperators(A)
ksp.setType(PETSc.KSP.Type.GMRES)
ksp.getPC().setType(PETSc.PC.Type.GAMG) # Using AMG as in your example

ksp.setFromOptions()
ksp.solve(b, u_sol)

print(f"Convergence Reason: {ksp.getConvergedReason()}")
print(f"Iterations = {ksp.getIterationNumber()}")

# --------------------------------------------------
# Visualization
# --------------------------------------------------
x_axis = np.linspace(0.0, 1.0, n)
u_final = u_sol.getArray()

plt.figure(figsize=(8, 5))
plt.plot(x_axis, u_final, 'o-', markersize=3, label='PETSc Steady State')
plt.title(f"Steady-State Convection-Diffusion (a={a}, nu={nu})")
plt.xlabel("x")
plt.ylabel("u")
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig("1d-steady-convection-diffusion.png")