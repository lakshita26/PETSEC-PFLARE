import numpy as np
import pflare  
from petsc4py import PETSc
import pflare_defs


def main():

    n = PETSc.Options().getInt("n", 100)
    second_solve = PETSc.Options().getBool("second_solve", False)

    comm = PETSc.COMM_WORLD

    # -------------------
    # GPU vectors
    # -------------------
    x = PETSc.Vec().create(comm=comm)
    x.setSizes(n)
    x.setType("cuda")   # or "kokkos"
    x.setFromOptions()

    b = x.duplicate()

    local_size = x.getLocalSize()
    rstart, rend = x.getOwnershipRange()

    # -------------------
    # GPU matrix
    # -------------------
    A = PETSc.Mat().create(comm=comm)
    A.setSizes([[local_size, n], [local_size, n]])
    A.setType("aijcusparse")  # GPU sparse matrix

    oor = np.zeros(2 * local_size, dtype=np.int32)
    ooc = np.zeros(2 * local_size, dtype=np.int32)

    counter = 0
    for i in range(rstart, rend):

        if i == 0:
            oor[counter] = -1
            ooc[counter] = -1
            oor[counter + 1] = 0
            ooc[counter + 1] = 0
        else:
            oor[counter] = i
            ooc[counter] = i - 1
            oor[counter + 1] = i
            ooc[counter + 1] = i

        counter += 2

    A.setPreallocationCOO(oor, ooc)

    vals = np.tile([-1.0, 1.0], local_size)
    A.setValuesCOO(vals)
    A.assemble()

    b.set(0.0)

    # -------------------
    # KSP solver
    # -------------------
    ksp = PETSc.KSP().create(comm=comm)
    ksp.setInitialGuessNonzero(True)
    ksp.setOperators(A)

    pc = ksp.getPC()

    # ✅ Use PCAIR (same as your C++ code)
    pc.setType("air")

    ksp.setFromOptions()
    ksp.setUp()

    # Warm-up solve (GPU staging like C++)
    x.set(1.0)
    ksp.solve(b, x)

    if second_solve:
        x.set(1.0)
        ksp.solve(b, x)

    PETSc.Sys.Print("Converged reason:", ksp.getConvergedReason())

    # Cleanup
    x.destroy()
    b.destroy()
    A.destroy()
    ksp.destroy()


if __name__ == "__main__":
    main()
