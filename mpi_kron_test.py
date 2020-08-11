import numpy as np
import scipy.sparse
from mpi4py import MPI

from mpi_kron import (IdentityKronMatMPI, MatKronIdentityMPI,
                      TridiagKronIdentityMPI, as_matrix)
from mpi_vector import KronVectorMPI


def test_identity_kron_mat():
    N = 13
    M = 16
    mat_space = np.arange(0, M * M).reshape(M, M)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Create some global vector on root.
    x_mpi = KronVectorMPI(comm, N, M)
    x_glob = None
    if rank == 0:
        x_glob = np.random.rand(N * M) * 1.0
        y_glob = np.kron(np.eye(N), mat_space) @ x_glob
    x_mpi.scatter(x_glob)

    # Apply the vector using MPI
    I_M = IdentityKronMatMPI(N, mat_space)
    x_mpi = I_M @ x_mpi

    comm.Barrier()
    # Check that it is corret.
    x_mpi.gather(x_glob)
    if rank == 0:
        print(x_glob.reshape(N, M))
        assert (np.allclose(y_glob, x_glob))


def test_mat_kron_identity():
    N = 9
    M = 16
    mat_time = np.arange(0, N * N).reshape(N, N)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Create some global vector on root.
    x_mpi = KronVectorMPI(comm, N, M)
    x_glob = None
    if rank == 0:
        x_glob = np.random.rand(N * M) * 1.0
        y_glob = np.kron(mat_time, np.eye(M)) @ x_glob
    x_mpi.scatter(x_glob)

    # Apply the vector using MPI
    M_I = MatKronIdentityMPI(mat_time, M)
    x_mpi = M_I @ x_mpi

    # Check that it is corret.
    x_mpi.gather(x_glob)
    if rank == 0:
        assert (np.allclose(y_glob, x_glob))


def test_tridiag_kron_mat():
    mat = np.array([[3.5, 13., 28.5, 50.,
                     77.5], [-5., -23., -53., -95., -149.],
                    [2.5, 11., 25.5, 46., 72.5]])
    stiff_time = scipy.sparse.spdiags(mat, (1, 0, -1), 5, 5).T.copy().tocsr()
    N = stiff_time.shape[0]
    M = 3
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Create some global vector on root.
    x_mpi = KronVectorMPI(comm, N, M)
    x_glob = None
    if rank == 0:
        x_glob = np.random.rand(N * M) * 1.0
        y_glob = np.kron(as_matrix(stiff_time), as_matrix(np.eye(M))) @ x_glob
    x_mpi.scatter(x_glob)

    # Apply the vector using MPI
    T_M = TridiagKronIdentityMPI(stiff_time, M)
    x_mpi = T_M @ x_mpi

    # Check that it is corret.
    x_mpi.gather(x_glob)
    if rank == 0:
        assert (np.allclose(y_glob, x_glob))
