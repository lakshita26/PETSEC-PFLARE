'''
Ex86 from PETSc example files implemented for PETSc4py.
Modified version of ex86 to use PFLARE and COO assembly.

Solves a one-dimensional steady upwind advection system with KSP.

Input parameters include:
    -n <mesh_n>         : number of mesh points
    -second_solve       : whether to run a second solve
    -ksp_monitor        : monitor the iterative solver
    -pc_type air        : use the AIR preconditioner (requires pflare)
'''
import sys
import petsc4py

petsc4py.init(sys.argv)
from petsc4py import PETSc

# Strictly import pflare. If this is missing, the script will error out, 
# just like the C code would fail to compile/link without "pflare.h".
import pflare 

import numpy as np

comm = PETSc.COMM_WORLD
size = comm.getSize()
rank = comm.getRank()

OptDB = PETSc.Options()
n = OptDB.getInt('n', 100)
second_solve = OptDB.getBool('second_solve', False)

# The C code calls PCRegister_PFLARE(). In Python, importing the module 
# usually triggers the registration automatically.
# If your pflare python module requires an explicit init, you would call it here.
# pflare.register() 

setup_stage = PETSc.Log.Stage("Setup")
gpu_copy_stage = PETSc.Log.Stage("GPU copy stage - triggered by a prelim KSPSolve")

if rank == 0:
    print(f"Starting Ex86 with n={n}...")

''' - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        Compute the matrix and right-hand-side vector
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - '''
x = PETSc.Vec().create(comm=comm)
x.setName("Solution")
x.setSizes(n)
x.setFromOptions()

b = x.duplicate()
local_size = x.getLocalSize()
global_row_start, global_row_end = x.getOwnershipRange()

A = PETSc.Mat().create(comm=comm)
A.setSizes((local_size, n))
A.setFromOptions()

# COO Assembly logic strictly matching C
rows = []
cols = []
vals = []

counter = 0
start_assign = global_row_start

if global_row_start == 0:
    start_assign = 1
    rows.append(0)
    cols.append(0)
    vals.append(1.0)
    counter += 1

for i in range(start_assign, global_row_end):
    # Upwind finite difference operator
    
    # i - 1 term
    rows.append(i)
    cols.append(i - 1)
    vals.append(-1.0)
    
    # i term
    rows.append(i)
    cols.append(i)
    vals.append(1.0)
    
    counter += 2

# Convert to numpy for PETSc call
np_rows = np.array(rows, dtype=np.int32)
np_cols = np.array(cols, dtype=np.int32)
np_vals = np.array(vals, dtype=PETSc.ScalarType)

A.setPreallocationCOO(np_rows, np_cols)
A.setValuesCOO(np_vals, PETSc.InsertMode.INSERT_VALUES)

# Zero RHS
b.set(0.0)

if rank == 0:
    print("Matrix assembled.")

''' - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            Create the linear solver and set various options
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - '''
ksp = PETSc.KSP().create(comm=comm)
ksp.setInitialGuessNonzero(True)
ksp.setOperators(A, A)

pc = ksp.getPC()

# Strictly set type to AIR (PCAIR). 
# If 'air' is not registered, this will raise a PETSc Error, matching C behavior.
pc.setType("air")

ksp.setFromOptions()

# Setup Stage
setup_stage.push()
ksp.setUp()
setup_stage.pop()

# Preliminary KSPSolve for GPU copy
x.set(1.0)

gpu_copy_stage.push()
ksp.solve(b, x)
gpu_copy_stage.pop()

''' - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    Solve the linear system
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - '''
if second_solve:
    if rank == 0:
        print("Running second solve...")
    x.set(1.0)
    ksp.solve(b, x)

reason = ksp.getConvergedReason()
iterations = ksp.getIterationNumber()

if rank == 0:
    print(f"Solver finished. Iterations: {iterations}")
    if reason > 0:
        print(f"Converged. Reason: {reason}")
    else:
        print(f"Diverged. Reason: {reason}")

if reason < 0:
    sys.exit(1)