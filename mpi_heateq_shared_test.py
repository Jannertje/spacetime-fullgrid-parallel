import numpy as np
import scipy.sparse
from mpi4py import MPI

from lanczos import Lanczos
from linalg import PCG
from linop import AsLinearOperator, CompositeLinOp
from mpi_heateq import HeatEquationMPI
from mpi_heateq_shared import HeatEquationMPIShared
from mpi_kron import (BlockDiagMPI, CompositeMPI, MatKronIdentityMPI, SumMPI,
                      TridiagKronMatMPI, as_matrix)
from mpi_kron_test import linearity_test_MPI, linop_test_MPI
from mpi_vector import KronVectorMPI
from problem import square


def test_shared_equals_normal():
    refines = 4
    heat_eq_mpi = HeatEquationMPI(refines, precond='mg')
    heat_eq_mpi_shared = HeatEquationMPIShared(refines)
    N = heat_eq_mpi.N
    M = heat_eq_mpi.M

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Solve using mpi.
    def cb(w, residual, k):
        if rank == 0:
            print('.', end='', flush=True)

    u_mpi, iters_mpi = PCG(heat_eq_mpi.WT_S_W,
                           heat_eq_mpi.P,
                           heat_eq_mpi.rhs,
                           callback=cb)
    u_mpi_shared, iters_mpi_shared = PCG(heat_eq_mpi_shared.WT_S_W,
                                         heat_eq_mpi_shared.P,
                                         heat_eq_mpi_shared.rhs,
                                         callback=cb)

    # Solve using demo on root.
    u_glob_mpi = None
    u_glob_mpi_shared = None
    if rank == 0:
        u_glob_mpi = np.empty(N * M)
        u_glob_mpi_shared = np.empty(N * M)

    # Gather the results on root.
    u_mpi.gather(u_glob_mpi)
    u_mpi_shared.gather(u_glob_mpi_shared)

    # Compare.
    if rank == 0:
        assert (np.linalg.norm(u_glob_mpi - u_glob_mpi_shared) < 1e-5)
