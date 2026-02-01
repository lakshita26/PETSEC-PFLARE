'''
DMDA/KSP solving a system of linear equations.
Steady advection-diffusion equation in 2D with finite difference.
Advection is upwinded.

Usage:
  python adv_diff_2d.py
      : pure advection with theta = pi/4, dimensionless
  python adv_diff_2d.py -adv_nondim 0
      : pure advection with theta = pi/4, scaled by Hx * Hy
  python adv_diff_2d.py -u 0 -v 0 -alpha 1.0 
      : pure diffusion scaled by Hx * Hy (Dirichlet all sides)
  python adv_diff_2d.py -alpha 1.0 
      : advection-diffusion scaled by Hx * Hy with theta=pi/4
'''

import sys
import petsc4py
import numpy as np

petsc4py.init(sys.argv)
from petsc4py import PETSc

# Import pflare if available
try:
    import pflare
except ImportError:
    pass 

comm = PETSc.COMM_WORLD
size = comm.getSize()
rank = comm.getRank()

OptDB = PETSc.Options()

# ~~~~~~~~~~~~~~
# Logging Stages
# ~~~~~~~~~~~~~~
setup_stage = PETSc.Log.Stage("Setup")
gpu_copy_stage = PETSc.Log.Stage("GPU copy stage - triggered by a prelim KSPSolve")

''' - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        Get Parameters and Options
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - '''

# Dimensions of box, L_y x L_x - default to [0, 1]^2
L_x = OptDB.getReal('L_x', 1.0)
L_y = OptDB.getReal('L_y', 1.0)

if L_x < 0.0: raise ValueError("L_x must be positive")
if L_y < 0.0: raise ValueError("L_y must be positive")

second_solve = OptDB.getBool('second_solve', False)

# Advection velocities
theta = OptDB.getReal('theta', np.pi / 4.0)
if not (0.0 <= theta <= np.pi / 2.0):
    raise ValueError("Theta must be between 0 and pi/2")

u = np.cos(theta)
v = np.sin(theta)

# Manual u/v overrides
if OptDB.hasName('u') or OptDB.hasName('v'):
    u_in = OptDB.getReal('u', u)
    v_in = OptDB.getReal('v', v)
    if u_in < 0.0 or v_in < 0.0:
        raise ValueError("u and v must be positive")
    u = u_in
    v = v_in

# Diffusion coefficient
alpha = OptDB.getReal('alpha', 0.0)

# Non-dimensional flag
# Default: True if pure advection (alpha=0), False if diffusion exists
default_adv_nondim = True if alpha == 0.0 else False
adv_nondim = OptDB.getBool('adv_nondim', default_adv_nondim)

if alpha != 0.0 and adv_nondim:
    raise ValueError("Non-dimensional advection only applies without diffusion")

# Diagonal Scaling flag
diag_scale = OptDB.getBool('diag_scale', False)

''' - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        Compute the matrix and right-hand-side vector that define
        the linear system, Ax = b.
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - '''
'''
    Create DMDA (Structured Grid).
    11x11 grid, 1 DOF per node, Stencil Width 1, Star Stencil.
    This replaces the manual MatCreate/SetSizes from Ex2 because 
    the C code relies on DMDA geometry.
'''
da = PETSc.DMDA().create(
    dim=2,
    dof=1,
    sizes=(11, 11),
    proc_sizes=(PETSc.DECIDE, PETSc.DECIDE),
    boundary_type=(PETSc.DM.BoundaryType.NONE, PETSc.DM.BoundaryType.NONE),
    stencil_type=PETSc.DMDA.StencilType.STAR,
    stencil_width=1,
    comm=comm
)
da.setFromOptions()
da.setUp()
da.setUniformCoordinates(xmin=0.0, xmax=L_x, ymin=0.0, ymax=L_y, zmin=0.0, zmax=0.0)

'''
    Create Matrix from DMDA.
    We set option to ignore zero entries to maintain sparsity pattern.
'''
A = da.createMatrix()
A.setOption(PETSc.Mat.Option.IGNORE_ZERO_ENTRIES, True)

'''
    Calculate Grid Info (Hx, Hy, scaling factors)
'''
(M, N) = da.getSizes()
Hx = L_x / M
Hy = L_y / N

HxdHy = Hx / Hy
HydHx = Hy / Hx

adv_x_scale = Hx
adv_y_scale = Hy

if adv_nondim:
    adv_x_scale = 1.0
    adv_y_scale = HydHx

'''
    Determine which rows of the matrix are locally owned.
    DMDAGetCorners returns the local (x, y) range for this processor.
'''
(xs, ys, zs, xm, ym, zm) = da.getCorners()

'''
    Set matrix elements for the 2-D Stencil.
    Loop over local nodes (y then x to match C ordering).
'''
for j in range(ys, ys + ym):
    for i in range(xs, xs + xm):
        
        row = PETSc.Mat.Stencil()
        row.index = (i, j)
        row.field = 0

        cols = []
        vals = []

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Boundary Conditions
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if i == 0 or j == 0 or i == M - 1 or j == N - 1:
            
            # Dirichlet on Left or Bottom
            if i == 0 or j == 0:
                c = PETSc.Mat.Stencil()
                c.index = (i, j); c.field = 0
                cols.append(c); vals.append(1.0)
            
            # Top or Right Boundary
            else:
                # Pure Advection (Outflow BC)
                if alpha == 0.0:
                    # Bottom neighbor (i, j-1)
                    c0 = PETSc.Mat.Stencil(); c0.index = (i, j-1); c0.field = 0
                    cols.append(c0); vals.append(-u * adv_y_scale)
                    
                    # Left neighbor (i-1, j)
                    c1 = PETSc.Mat.Stencil(); c1.index = (i-1, j); c1.field = 0
                    cols.append(c1); vals.append(-v * adv_x_scale)
                    
                    # Center (i, j)
                    c2 = PETSc.Mat.Stencil(); c2.index = (i, j); c2.field = 0
                    cols.append(c2); vals.append(u * adv_y_scale + v * adv_x_scale)

                # Diffusion exists (Dirichlet BC on Top/Right)
                else:
                    c = PETSc.Mat.Stencil(); c.index = (i, j); c.field = 0
                    cols.append(c); vals.append(1.0)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Interior Nodes
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        else:
            # 1. Add Diffusion Terms
            if alpha != 0.0:
                # Bottom (i, j-1)
                c0 = PETSc.Mat.Stencil(); c0.index = (i, j-1); c0.field = 0
                cols.append(c0); vals.append(-alpha * HxdHy)
                
                # Left (i-1, j)
                c1 = PETSc.Mat.Stencil(); c1.index = (i-1, j); c1.field = 0
                cols.append(c1); vals.append(-alpha * HydHx)
                
                # Center (i, j)
                c2 = PETSc.Mat.Stencil(); c2.index = (i, j); c2.field = 0
                cols.append(c2); vals.append(alpha * 2.0 * (HxdHy + HydHx))
                
                # Right (i+1, j)
                c3 = PETSc.Mat.Stencil(); c3.index = (i+1, j); c3.field = 0
                cols.append(c3); vals.append(-alpha * HydHx)
                
                # Top (i, j+1)
                c4 = PETSc.Mat.Stencil(); c4.index = (i, j+1); c4.field = 0
                cols.append(c4); vals.append(-alpha * HxdHy)

            # 2. Add Advection Terms
            if u != 0.0 or v != 0.0:
                # Bottom (i, j-1)
                c0 = PETSc.Mat.Stencil(); c0.index = (i, j-1); c0.field = 0
                cols.append(c0); vals.append(-u * adv_y_scale)
                
                # Left (i-1, j)
                c1 = PETSc.Mat.Stencil(); c1.index = (i-1, j); c1.field = 0
                cols.append(c1); vals.append(-v * adv_x_scale)
                
                # Center (i, j)
                c2 = PETSc.Mat.Stencil(); c2.index = (i, j); c2.field = 0
                cols.append(c2); vals.append(u * adv_y_scale + v * adv_x_scale)

        # Set values for this row
        A.setValuesStencil(row, cols, vals, PETSc.InsertMode.ADD_VALUES)

'''
    Assemble matrix.
'''
A.assemblyBegin(PETSc.Mat.AssemblyType.FINAL)
A.assemblyEnd(PETSc.Mat.AssemblyType.FINAL)

# Compress matrix memory (matches MatDuplicate logic in C)
A_temp = A.duplicate(copy=True)
A.destroy()
A = A_temp

'''
    Create parallel vectors.
'''
x = da.createGlobalVec()
b = da.createGlobalVec()

# Zero RHS
b.set(0.0)

# Diagonal Scaling (if enabled)
if diag_scale:
    diag_vec = x.duplicate()
    A.getDiagonal(diag_vec)
    diag_vec.reciprocal()
    
    # Scale Matrix: D^{-1} A
    A.diagonalScale(left=diag_vec, right=None)
    
    # Scale RHS: D^{-1} b
    b.pointwiseMult(diag_vec, b)
    diag_vec.destroy()

''' - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            Create the linear solver and set various options
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - '''
ksp = PETSc.KSP().create(comm=comm)

# We generate the matrix ourselves, so disable standard DM KSP matrix generation
ksp.setDM(da)
ksp.setDMActive(False)

'''
    Set operators.
'''
ksp.setOperators(A, A)
ksp.setInitialGuessNonzero(True)

# ~~~~~~~~~~~~~~
# Let's use AIR
# ~~~~~~~~~~~~~~

pc = ksp.getPC()
# Explicitly set PC type to AIR (PFLARE)
try:
    pc.setType("air")
except PETSc.Error:
    pass

# ~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~

ksp.setFromOptions()

# Setup Stage
setup_stage.push()
ksp.setUp()
setup_stage.pop()

# Preliminary Solve (GPU Copy Stage)
x.set(1.0)
gpu_copy_stage.push()
ksp.solve(b, x)
gpu_copy_stage.pop()

''' - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    Solve the linear system
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - '''

if second_solve:
    x.set(1.0)
    ksp.solve(b, x)

''' - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    Check the solution and clean up
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - '''
reason = ksp.getConvergedReason()
iterations = ksp.getIterationNumber()

if rank == 0:
    # Optional print to verify execution
    # print(f"Solver finished. Iterations: {iterations}, Reason: {reason}")
    pass

da.destroy()
ksp.destroy()
x.destroy()
b.destroy()
A.destroy()

if reason < 0:
    sys.exit(1)