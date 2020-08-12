import numpy as np
import scipy.sparse
from mpi4py import MPI

from demo import demo
from lanczos import Lanczos
from linalg import PCG
from mpi_heateq import HeatEquationMPI
from mpi_kron import IdentityMPI, as_matrix
from mpi_kron_test import linearity_test_MPI, linop_test_MPI
from mpi_vector import KronVectorMPI
from problem import square

refines = 2


def test_bilforms():
    def test_linop(N, M, linop):
        comm = MPI.COMM_WORLD
        # Create some global vector on root.
        x_mpi = KronVectorMPI(comm, N, M)
        x_glob = None
        if comm.Get_rank() == 0:
            x_glob = np.random.rand(N * M) * 1.0
            y_glob = np.kron(as_matrix(linop.mat_time),
                             as_matrix(linop.mat_space)) @ x_glob
        x_mpi.scatter(x_glob)

        # Apply the vector using MPI
        x_mpi = linop @ x_mpi

        # Check that it is corret.
        x_mpi.gather(x_glob)
        if comm.Get_rank() == 0:
            assert (np.allclose(y_glob, x_glob))

    # Gather the space/time stiffness matrices.
    refines = 2
    heat_eq_mpi = HeatEquationMPI(refines)
    N = heat_eq_mpi.N
    M = heat_eq_mpi.M
    for linop in heat_eq_mpi.S.linops:
        test_linop(N, M, linop)


def test_S_apply():
    # Gather the space/time stiffness matrices.
    refines = 2
    heat_eq_mpi = HeatEquationMPI(refines)
    N = heat_eq_mpi.N
    M = heat_eq_mpi.M

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Create some global vector on root.
    x_mpi = KronVectorMPI(comm, N, M)
    x_glob = None
    if rank == 0:
        np.random.seed(0)
        x_glob = np.random.rand(N * M) * 1.0

        # Compare to demo
        _, _, _, S, _, _, _, _, _, _, _ = demo(*square(refines))
        y_glob = S @ x_glob

        # Compare to np.kron
        S = sum([linop.as_matrix() for linop in heat_eq_mpi.S.linops])
        z_glob = S @ x_glob

    # And apply it using MPI :-)
    x_mpi.scatter(x_glob)
    y_mpi = heat_eq_mpi.S @ x_mpi
    y_mpi.gather(x_glob)
    if rank == 0:
        assert (np.allclose(x_glob, y_glob))
        assert (np.allclose(x_glob, z_glob))


def test_solve():
    refines = 2
    heat_eq_mpi = HeatEquationMPI(refines)
    N = heat_eq_mpi.N
    M = heat_eq_mpi.M

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Solve using demo on root.
    u_glob_mpi = None
    f_glob_mpi = None
    if rank == 0:
        u_glob_mpi = np.empty(N * M)
        f_glob_mpi = np.empty(N * M)

        # Extract f_glob from demo.
        _, _, _, S, _, _, _, _, f_glob_demo, _, _ = demo(*square(refines))

        # Solve on root.
        u_glob_demo = PCG(S, scipy.sparse.identity(N * M), f_glob_demo)

    # Solve using mpi.
    def cb(w, residual, k):
        if rank == 0:
            print('.', end='', flush=True)

    u_mpi = PCG(heat_eq_mpi.S, IdentityMPI(N, M), heat_eq_mpi.rhs, callback=cb)

    # Gather the results on root.
    u_mpi.gather(u_glob_mpi)

    # Compare to demo.
    heat_eq_mpi.rhs.gather(f_glob_mpi)
    if rank == 0:
        assert (np.allclose(f_glob_demo, f_glob_mpi))
        assert (np.allclose(u_glob_demo, u_glob_mpi))


def test_demo():
    refines = 4
    heat_eq_mpi = HeatEquationMPI(refines)
    print(heat_eq_mpi.W.N, heat_eq_mpi.W.M)
    linearity_test_MPI(heat_eq_mpi.W)

    _, _, WT, S, W, _, P, _, _, _, _ = demo(*square(refines))
    linop_test_MPI(heat_eq_mpi.W, as_matrix(W))
    linop_test_MPI(heat_eq_mpi.WT, as_matrix(WT))
    linop_test_MPI(heat_eq_mpi.S, as_matrix(S))
    linop_test_MPI(heat_eq_mpi.WT_S_W, as_matrix(WT @ S @ W))
    linop_test_MPI(heat_eq_mpi.P, as_matrix(P))


def test_preconditioner():
    refines = 4
    heat_eq_mpi = HeatEquationMPI(refines)
    N = heat_eq_mpi.N
    M = heat_eq_mpi.M
    comm = MPI.COMM_WORLD

    # Create random MPI vector.
    w_mpi = KronVectorMPI(comm, N, M)
    w_mpi.X_loc = np.random.rand(w_mpi.X_loc.shape[0], M)

    # Perform Lanczos.
    lanczos = Lanczos(heat_eq_mpi.WT_S_W, heat_eq_mpi.P, w=w_mpi)

    if w_mpi.rank == 0:
        print(N * M, lanczos)

        # Compare to demo
        #_, _, WT, S, W, _, P, _, _, _, _ = demo(*square(refines))
        #print('cond(P @ WT @ S @ W)', Lanczos(WT @ S @ W, P))
