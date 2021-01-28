# A parallel algorithm for solving linear parabolic equations
This repository supplements arXiv:2009.08875.

We implemented our algorithm in Python3 using the open source finite element
library NGSolve for meshing and descretization of the bilinear forms in space
and time, MPI through mpi4py for distributed computations, and SciPy for the
sparse matrix-vector computations.

## Requirements
- NGsolve, follow the NGSolve install instructions from https://ngsolve.org/docu/latest/install/install_sources.html.
- Python pacakges, see requirement.txt: mpi4py, numpy, pets4py, psutil, scipy.

## Instructions
A normal, single-threaded, implementation is given in heateq.py.
```bash
python3 heateq.py --J_time=3 --J_space=6 --problem=square
```

The results in the paper are gathered using the parallel implementation
given in heateq_mpi.py. Similar arguments hold, e.g.
```bash
mpirun -np 2 python3 heateq_mpi.py --J_time=3 --J_space=6 --problem=square
```

The tests can be run using pytest, e.g.
```bash
pytest
mpirun -np 2 pytest
```
