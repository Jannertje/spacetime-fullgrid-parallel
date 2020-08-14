import numpy as np
import scipy.sparse

from mpi4py import MPI
from mpi_kron import (BlockDiagMPI, IdentityKronMatMPI, LinearOperatorMPI,
                      MatKronIdentityMPI, TridiagKronIdentityMPI, as_matrix)
from mpi_vector import KronVectorMPI


def linearity_test_MPI(linop):
    assert isinstance(linop, LinearOperatorMPI)
    alpha = 3.14
    x_mpi = KronVectorMPI(MPI.COMM_WORLD, linop.N, linop.M)
    x_mpi.X_loc = np.random.rand(*x_mpi.X_loc.shape)

    y_mpi = KronVectorMPI(MPI.COMM_WORLD, linop.N, linop.M)
    y_mpi.X_loc = np.random.rand(*y_mpi.X_loc.shape)

    z_mpi = x_mpi + alpha * y_mpi

    result_1 = linop @ x_mpi + alpha * (linop @ y_mpi)
    result_2 = linop @ z_mpi

    assert (np.allclose(result_1.X_loc.reshape(-1),
                        result_2.X_loc.reshape(-1)))


def linop_test_MPI(linop_mpi, mat_glob):
    linearity_test_MPI(linop_mpi)
    rank = MPI.COMM_WORLD.Get_rank()
    mat_mpi = linop_mpi.as_global_matrix()
    if rank == 0:
        if not np.allclose(mat_mpi, mat_glob):
            import matplotlib.pyplot as plt
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
            ax1.imshow(np.log10(np.abs(mat_mpi)))
            ax2.imshow(np.log10(np.abs(mat_glob)))
            ax3.imshow(np.log10(np.abs(mat_mpi - mat_glob)))
            plt.show()
        assert (np.allclose(mat_mpi, mat_glob))


def test_identity_kron_mat():
    N = 13
    M = 16
    mat_space = np.arange(0, M * M).reshape(M, M)
    I_M = IdentityKronMatMPI(N, mat_space)
    linop_test_MPI(I_M, np.kron(np.eye(N), mat_space))


def test_mat_kron_identity():
    N = 9
    M = 16
    mat_time = np.arange(0, N * N).reshape(N, N)
    M_I = MatKronIdentityMPI(mat_time, M)
    linop_test_MPI(M_I, np.kron(mat_time, np.eye(M)))


def test_tridiag_kron_mat():
    mat = np.array([[3.5, 13., 28.5, 50.,
                     77.5], [-5., -23., -53., -95., -149.],
                    [2.5, 11., 25.5, 46., 72.5]])
    stiff_time = scipy.sparse.spdiags(mat, (1, 0, -1), 5, 5).T.copy().tocsr()

    M = 3
    T_M = TridiagKronIdentityMPI(stiff_time, M)
    linop_test_MPI(T_M, np.kron(stiff_time.toarray(), np.eye(M)))


def test_block_diag():
    N = 9
    M = 16
    matrices_space = []
    np.random.seed(0)
    for n in range(N):
        matrices_space.append(np.random.rand(M, M))

    Blk = BlockDiagMPI(matrices_space)
    linop_test_MPI(Blk, scipy.sparse.block_diag(matrices_space).toarray())
