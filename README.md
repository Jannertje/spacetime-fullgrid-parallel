# A scalable algorithm for solving linear parabolic equations
This repository supplements arXiv:2009.08875.

We implemented our algorithm in Python using the open source finite element
library NGSolve for meshing and descretization of the bilinear forms in space
and time, MPI through mpi4py for distributed computations, and SciPy for the
sparse matrix-vector computations.

## Requirements
- SciPy, version >= 1.4.0.
- NGsolve, follow the NGSolve install instructions from https://ngsolve.org/docu/latest/install/install_sources.html.

## Instructions
Run mpi_heateq_shared.py
