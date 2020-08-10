from pprint import pprint

import numpy as np
import scipy
from mpi4py import MPI
from ngsolve import H1, L2, BilinearForm, InnerProduct, dx, grad
from scipy.sparse.linalg.interface import LinearOperator

from bilform import KronBF
from fespace import KronFES
from mpi_kron import (IdentityKronMatMPI, TridiagKronIdentityMPI,
                      TridiagKronMatMPI)
from mpi_vector import VectorTimeMPI
from problem import cube, square


def as_matrix(operator):
    if hasattr(operator, "A"):
        return operator.A
    cols = operator.shape[1]
    return operator @ np.eye(cols)


class IdentityKronMat(LinearOperator):
    def __init__(self, size_time, mat_space):
        self.N = size_time
        self.M, L = mat_space.shape
        assert (self.M == L)
        self.mat_space = mat_space
        super().__init__(dtype=np.float64,
                         shape=(self.N * self.M, self.N * self.M))
        # Sanity check.
        assert (np.allclose(as_matrix(self),
                            np.kron(np.eye(size_time), as_matrix(mat_space))))

    def _matvec(self, x):
        return (self.mat_space @ x.reshape(self.N, self.M).T).T


class MatKronIdentity(LinearOperator):
    def __init__(self, mat_time, bandwith_time, size_space):
        self.N, K = mat_time.shape
        self.M = size_space
        self.mat_time = mat_time
        assert (self.N == K)
        super().__init__(dtype=np.float64,
                         shape=(self.N * self.M, self.N * self.M))
        # Check that it is indeed bandwidthed
        self.B = bandwith_time
        for t, row in enumerate(mat_time):
            assert np.all(row.indices == np.arange(max(
                0, t - self.B), min(t + self.B + 1, self.N)))
        # Sanity check.
        assert (np.allclose(as_matrix(self),
                            np.kron(as_matrix(mat_time), np.eye(size_space))))

    def _matvec(self, x):
        B = self.B
        X = x.reshape(self.N, self.M)
        Y = np.empty(X.shape, dtype=np.float64)

        for t, row in enumerate(self.mat_time):
            assert np.all(row.indices >= t - B)
            assert np.all(row.indices <= t + B)
            Y[t, :] = row[0].data @ X[max(0, t - B):min(t + B + 1, self.N), :]
        return Y.reshape(-1)


def stiff_kron(N=2):
    order = 1
    mesh_space, bc_space, mesh_time, data, fn = square(nrefines=N)
    X = KronFES(H1(mesh_time, order=order),
                H1(mesh_space, order=order, dirichlet=bc_space))
    stiff_bf = KronBF(X, X, lambda u, v: InnerProduct(grad(u), grad(v)) * dx,
                      lambda u, v: InnerProduct(grad(u), grad(v)) * dx)
    linop = stiff_bf.assemble()
    stiff_space = stiff_bf.space.mat
    stiff_time = stiff_bf.time.mat
    # Sanity check.
    assert (np.allclose(as_matrix(linop),
                        np.kron(as_matrix(stiff_time),
                                as_matrix(stiff_space))))
    return linop, stiff_time, stiff_space


# Gather the space/time stiffness matrices.
linop, stiff_time, stiff_space = stiff_kron()
N = stiff_time.shape[0]
M = stiff_space.shape[0]

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
print("Hello! I'm rank {} from {}. Stiff_time: {}".format(
    comm.rank, comm.size, stiff_time.data))

# Create some global vector on root.
x_mpi = VectorTimeMPI(comm, N, M)
x_glob = None
if rank == 0:
    x_glob = np.random.rand(N * M) * 1.0
    y_glob = np.kron(as_matrix(stiff_time), as_matrix(stiff_space)) @ x_glob
x_mpi.scatter(x_glob)

# Apply the vector using MPI
T_M = TridiagKronMatMPI(stiff_time, stiff_space)
T_M.matvec_inplace(x_mpi)

# Check that it is corret.
x_mpi.gather(x_glob)
if rank == 0:
    assert (np.allclose(y_glob, x_glob))
