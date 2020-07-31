# Parallel space-time fullgrid solution to the heat equation

## Instructions:
follow the NGSolve install instructions from https://ngsolve.org/docu/latest/install/install_sources.html
with extra option `-DUSE_MPI=ON -DUSE_MPI4PY=ON`. Later on, we will want to compile also with
`-DUSE_HYPRE=ON`, because we want to use the scalable HYPRE preconditioner
for shared-memory parallellism in space. I think the current build may be broken,
but they are actively pushing stuff with the `mpi4py` library.

The serial code can be run with `ngspy demo.py` (basically `python` with some
extra hooks) or `netgen demo.py` (the entire NGSolve gui). MPI code can be run
with `mpiexec -np X ngspy demo.py`.
