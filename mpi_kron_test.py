import numpy as np
import scipy.sparse

from mpi4py import MPI
from mpi_kron import (BlockDiagMPI, CompositeMPI, IdentityKronMatMPI,
                      LinearOperatorMPI, MatKronIdentityMPI,
                      SparseKronIdentityMPI, TridiagKronIdentityMPI, as_matrix)
from mpi_vector import KronVectorMPI, DofDistributionMPI


def linearity_test_MPI(linop):
    assert isinstance(linop, LinearOperatorMPI)
    alpha = 3.14
    dofs_distr = DofDistributionMPI(MPI.COMM_WORLD, linop.N, linop.M)
    x_mpi = KronVectorMPI(dofs_distr)
    x_mpi.X_loc = np.random.rand(*x_mpi.X_loc.shape)

    y_mpi = KronVectorMPI(dofs_distr)
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
        assert np.allclose(mat_mpi, mat_glob)


def test_identity_kron_mat():
    N = 13
    M = 16
    dofs_distr = DofDistributionMPI(MPI.COMM_WORLD, N, M)
    mat_space = np.arange(0, M * M).reshape(M, M)
    I_M = IdentityKronMatMPI(dofs_distr, mat_space)
    linop_test_MPI(I_M, np.kron(np.eye(N), mat_space))


def test_mat_kron_identity():
    N = 9
    M = 16
    dofs_distr = DofDistributionMPI(MPI.COMM_WORLD, N, M)
    mat_time = np.arange(0, N * N).reshape(N, N)
    M_I = MatKronIdentityMPI(dofs_distr, mat_time)
    linop_test_MPI(M_I, np.kron(mat_time, np.eye(M)))


def test_tridiag_kron_mat():
    mat = np.array([[3.5, 13., 28.5, 50.,
                     77.5], [-5., -23., -53., -95., -149.],
                    [2.5, 11., 25.5, 46., 72.5]])
    stiff_time = scipy.sparse.spdiags(mat, (1, 0, -1), 5, 5).T.copy().tocsr()

    M = 3
    dofs_distr = DofDistributionMPI(MPI.COMM_WORLD, stiff_time.shape[0], M)
    T_M = TridiagKronIdentityMPI(dofs_distr, stiff_time)
    linop_test_MPI(T_M, np.kron(stiff_time.toarray(), np.eye(M)))


def test_sparse_kron_mat():
    mat = np.array([[3.5, 13., 28.5, 50.,
                     77.5], [-5., -23., -53., -95., -149.],
                    [2.5, 11., 25.5, 46., 72.5]])
    stiff_time = scipy.sparse.spdiags(mat, (1, 0, -1), 5, 5).T.copy().tocsr()

    M = 3
    dofs_distr = DofDistributionMPI(MPI.COMM_WORLD, stiff_time.shape[0], M)
    T_M = SparseKronIdentityMPI(dofs_distr, stiff_time)
    linop_test_MPI(T_M, np.kron(stiff_time.toarray(), np.eye(M)))


#
def test_block_diag():
    N = 9
    M = 16
    dofs_distr = DofDistributionMPI(MPI.COMM_WORLD, N, M)
    matrices_space = []
    np.random.seed(0)
    for n in range(N):
        matrices_space.append(np.random.rand(M, M))

    Blk = BlockDiagMPI(dofs_distr, matrices_space)
    linop_test_MPI(Blk, scipy.sparse.block_diag(matrices_space).toarray())


#
#
def test_composite():
    N = 9
    M = 16
    dofs_distr = DofDistributionMPI(MPI.COMM_WORLD, N, M)
    mat_time = np.arange(0, N * N).reshape(N, N)
    mat_space = np.arange(0, M * M).reshape(M, M)
    M_I = MatKronIdentityMPI(dofs_distr, mat_time)
    I_M = IdentityKronMatMPI(dofs_distr, mat_space)
    linop = CompositeMPI(dofs_distr, [I_M, M_I])
    composite_mat = linop.as_global_matrix()
    rank = MPI.COMM_WORLD.Get_rank()
    if rank == 0:
        assert np.allclose(composite_mat, np.kron(mat_time, mat_space))
