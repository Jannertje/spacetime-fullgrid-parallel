from pprint import pprint

import numpy as np
import scipy
import scipy.sparse
from mpi4py import MPI
from ngsolve import (H1, L2, BilinearForm, InnerProduct, Preconditioner, ds,
                     dx, grad)
from scipy.sparse.linalg.interface import LinearOperator

from bilform import BilForm, KronBF
from demo import demo
from fespace import KronFES
from linop import AsLinearOperator, CompositeLinOp
from mpi_kron import (IdentityKronMatMPI, TridiagKronIdentityMPI,
                      TridiagKronMatMPI, as_matrix)
from mpi_vector import VectorTimeMPI
from problem import cube, square


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


class HeatEquationMPI:
    def __init__(self, refines=2, precond='multigrid', order=1):
        mesh_space, bc_space, mesh_time, data, fn = square(nrefines=refines)
        X = KronFES(H1(mesh_time, order=order),
                    H1(mesh_space, order=order, dirichlet=bc_space))
        self.N = len(X.time.fd)
        self.M = len(X.space.fd)

        # --- TIME ---
        self.A_t = BilForm(
            X.time,
            bilform_lambda=lambda u, v: grad(u) * grad(v) * dx).assemble()
        self.L_t = BilForm(
            X.time, bilform_lambda=lambda u, v: u * grad(v) * dx).assemble()
        self.M_t = BilForm(X.time,
                           bilform_lambda=lambda u, v: u * v * dx).assemble()
        self.G_t = BilForm(
            X.time,
            bilform_lambda=lambda u, v: u * v * ds('start')).assemble()

        # --- SPACE ---
        self.M_x = BilForm(X.space,
                           bilform_lambda=lambda u, v: u * v * dx).assemble()

        # Stiffness + multigrid preconditioner space.
        K_x = BilForm(
            X.space,
            bilform_lambda=lambda u, v: InnerProduct(grad(u), grad(v)) * dx)
        Kinv_x_pc = Preconditioner(K_x.bf, precond)
        self.A_x = K_x.assemble()
        self.Kinv_x = AsLinearOperator(Kinv_x_pc.mat, X.space.fd)

        # -- MPI objects --
        self.A_MKM = TridiagKronMatMPI(
            self.A_t, CompositeLinOp([self.M_x, self.Kinv_x, self.M_x]))
        self.L_MKA = TridiagKronMatMPI(
            self.L_t, CompositeLinOp([self.M_x, self.Kinv_x, self.A_x]))
        self.LT_AKM = TridiagKronMatMPI(
            self.L_t.T.tocsr(),
            CompositeLinOp([self.A_x, self.Kinv_x, self.M_x]))
        self.M_AKA = TridiagKronMatMPI(
            self.M_t, CompositeLinOp([self.A_x, self.Kinv_x, self.A_x]))
        self.G_M = TridiagKronMatMPI(self.G_t, self.M_x)
        self.linops = [
            self.A_MKM, self.L_MKA, self.LT_AKM, self.M_AKA, self.G_M
        ]

    def matvec_inplace(self, vec):
        X_loc = vec.X_loc.copy()
        Y_loc = np.zeros(vec.X_loc.shape)

        for linop in self.linops:
            vec.X_loc[:] = X_loc
            linop.matvec_inplace(vec)
            Y_loc += vec.X_loc

        vec.X_loc[:] = Y_loc


def test_linop(N, M, linop):
    comm = MPI.COMM_WORLD
    # Create some global vector on root.
    x_mpi = VectorTimeMPI(comm, N, M)
    x_glob = None
    if rank == 0:
        x_glob = np.random.rand(N * M) * 1.0
        y_glob = np.kron(as_matrix(linop.mat_time), as_matrix(
            linop.mat_space)) @ x_glob
    x_mpi.scatter(x_glob)

    # Apply the vector using MPI
    linop.matvec_inplace(x_mpi)

    # Check that it is corret.
    x_mpi.gather(x_glob)
    if rank == 0:
        assert (np.allclose(y_glob, x_glob))


# Gather the space/time stiffness matrices.
refines = 2
heat_eq_mpi = HeatEquationMPI(refines)
N = heat_eq_mpi.N
M = heat_eq_mpi.M

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
print("Hello! I'm rank {} from {}.".format(comm.rank, comm.size))

test_linop(N, M, heat_eq_mpi.A_MKM)
test_linop(N, M, heat_eq_mpi.L_MKA)
test_linop(N, M, heat_eq_mpi.LT_AKM)
test_linop(N, M, heat_eq_mpi.G_M)

# Create some global vector on root.
x_mpi = VectorTimeMPI(comm, N, M)
x_glob = None
if rank == 0:
    np.random.seed(0)
    x_glob = np.random.rand(N * M) * 1.0
    # Compare to demo
    #_, _, _, S, _, _, _, _, _, _, _ = demo(*square(refines))
    #y_glob = S @ x_glob
    #print('y_glob', y_glob)
    S = sum([linop.as_matrix() for linop in heat_eq_mpi.linops])
    z_glob = S @ x_glob
    print('z_glob', z_glob)

# And apply it using MPI :-)
x_mpi.scatter(x_glob)

heat_eq_mpi.matvec_inplace(x_mpi)

x_mpi.gather(x_glob)
if rank == 0:
    assert (np.allclose(x_glob, z_glob))
