# A parallel algorithm for solving linear parabolic equations
This repository supplements arXiv:2009.08875.

We implemented our algorithm in Python3 using the open source finite element
library NGSolve for meshing and descretization of the bilinear forms in space
and time, MPI through mpi4py for distributed computations, and SciPy for the
sparse matrix-vector computations.

## Requirements
- NGsolve (tested with v6.2.2): [see install instructions](https://ngsolve.org/downloads).
- Python pacakges (requirements.txt): mpi4py, numpy, pets4py, psutil, scipy.

## Run instructions
A normal, single-threaded, implementation is given in heateq.py.
```bash
python3 heateq.py --J_time=3 --J_space=6 --problem=square
```

The parallel MPI implementation, used for the numerical results in the paper,
is given in heateq_mpi.py.
```bash
mpirun -np 2 python3 heateq_mpi.py --J_time=3 --J_space=6 --problem=square
```

The tests can be run using pytest.
```bash
pytest
mpirun -np 2 pytest
```
