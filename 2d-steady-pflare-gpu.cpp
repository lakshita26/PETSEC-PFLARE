#include <petscksp.h>
#include <Kokkos_Core.hpp>

int main(int argc, char **argv)
{
    PetscInitialize(&argc, &argv, NULL, NULL);
    Kokkos::initialize(argc, argv);

    MPI_Comm comm = PETSC_COMM_WORLD;

    int nx = 100, ny = 100;
    PetscOptionsGetInt(NULL, NULL, "-nx", &nx, NULL);
    PetscOptionsGetInt(NULL, NULL, "-ny", &ny, NULL);

    PetscPrintf(comm, "\n--- Experiment setup ---\n");
    PetscPrintf(comm, "Grid: %d x %d\n", nx, ny);

    // -----------------------------
    // DMDA grid
    // -----------------------------
    DM da;
    DMDACreate2d(comm,
                 DM_BOUNDARY_NONE,
                 DM_BOUNDARY_NONE,
                 DMDA_STENCIL_STAR,
                 nx, ny,
                 PETSC_DECIDE, PETSC_DECIDE,
                 1, 1,
                 NULL, NULL,
                 &da);

    DMSetUp(da);

    Mat A;
    Vec b, u;

    DMCreateMatrix(da, &A);
    DMCreateGlobalVector(da, &b);
    VecDuplicate(b, &u);

    // -----------------------------
    // Assembly (Kokkos parallel)
    // -----------------------------
    PetscInt xs, ys, xm, ym;
    DMDAGetCorners(da, &xs, &ys, NULL, &xm, &ym, NULL);

    Kokkos::parallel_for(
        "assemble",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
            {xs, ys}, {xs + xm, ys + ym}),
        KOKKOS_LAMBDA(int i, int j)
        {
            MatStencil row, col;

            row.i = i;
            row.j = j;

            // diagonal
            MatSetValueStencil(A, &row, &row, 2.0, INSERT_VALUES);

            if (i > 0) {
                col.i = i - 1;
                col.j = j;
                MatSetValueStencil(A, &row, &col, -1.0, INSERT_VALUES);
            }

            if (j > 0) {
                col.i = i;
                col.j = j - 1;
                MatSetValueStencil(A, &row, &col, -1.0, INSERT_VALUES);
            }
        });

    VecSet(b, 1.0);

    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);

    VecAssemblyBegin(b);
    VecAssemblyEnd(b);

    // -----------------------------
    // Solver
    // -----------------------------
    KSP ksp;
    KSPCreate(comm, &ksp);
    KSPSetOperators(ksp, A, A);
    KSPSetType(ksp, KSPGMRES);

    PC pc;
    KSPGetPC(ksp, &pc);
    PCSetType(pc, "air");

    KSPSetFromOptions(ksp);

    PetscPrintf(comm, "\n--- Solving ---\n");

    KSPSolve(ksp, b, u);

    PetscInt its;
    PetscReal res;

    KSPGetIterationNumber(ksp, &its);
    KSPGetResidualNorm(ksp, &res);

    PetscPrintf(comm, "\n--- Solver results ---\n");
    PetscPrintf(comm, "Iterations: %d\n", its);
    PetscPrintf(comm, "Residual: %e\n", res);

    // Cleanup
    KSPDestroy(&ksp);
    VecDestroy(&b);
    VecDestroy(&u);
    MatDestroy(&A);
    DMDestroy(&da);

    Kokkos::finalize();
    PetscFinalize();
    return 0;
}
