import numpy as np
from mpi4py import MPI

from demo import demo
from mpi_heateq import HeatEquationMPI
from mpi_kron import as_matrix
from mpi_vector import VectorTimeMPI
from problem import square

refines = 2


def test_bilforms():
    def test_linop(N, M, linop):
        comm = MPI.COMM_WORLD
        # Create some global vector on root.
        x_mpi = VectorTimeMPI(comm, N, M)
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
    for linop in heat_eq_mpi.linops:
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
    x_mpi = VectorTimeMPI(comm, N, M)
    x_glob = None
    if rank == 0:
        np.random.seed(0)
        x_glob = np.random.rand(N * M) * 1.0

        # Compare to demo
        _, _, _, S, _, _, _, _, _, _, _ = demo(*square(refines))
        y_glob = S @ x_glob

        # Compare to np.kron
        S = sum([linop.as_matrix() for linop in heat_eq_mpi.linops])
        z_glob = S @ x_glob

    # And apply it using MPI :-)
    x_mpi.scatter(x_glob)
    y_mpi = heat_eq_mpi @ x_mpi
    y_mpi.gather(x_glob)
    if rank == 0:
        assert (np.allclose(x_glob, y_glob))
        assert (np.allclose(x_glob, z_glob))
