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
from mpi_kron import (IdentityKronMatMPI, LinearOperatorMPI,
                      TridiagKronIdentityMPI, TridiagKronMatMPI, as_matrix)
from mpi_vector import VectorTimeMPI
from problem import cube, square


class HeatEquationMPI(LinearOperatorMPI):
    def __init__(self, refines=2, precond='multigrid', order=1):
        mesh_space, bc_space, mesh_time, data, fn = square(nrefines=refines)
        X = KronFES(H1(mesh_time, order=order),
                    H1(mesh_space, order=order, dirichlet=bc_space))
        super().__init__(len(X.time.fd), len(X.space.fd))

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

    def matvec(self, vec_in, vec_out):
        assert (vec_in is not vec_out)
        Y_loc = np.zeros(vec_out.X_loc.shape)

        for linop in self.linops:
            linop.matvec(vec_in, vec_out)
            Y_loc += vec_out.X_loc

        vec_out.X_loc = Y_loc
        return vec_out


if __name__ == "__main__":

    def test_linop(N, M, linop):
        comm = MPI.COMM_WORLD
        # Create some global vector on root.
        x_mpi = VectorTimeMPI(comm, N, M)
        x_glob = None
        if rank == 0:
            x_glob = np.random.rand(N * M) * 1.0
            y_glob = np.kron(as_matrix(linop.mat_time),
                             as_matrix(linop.mat_space)) @ x_glob
        x_mpi.scatter(x_glob)

        # Apply the vector using MPI
        x_mpi = linop @ x_mpi

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

    y_mpi = heat_eq_mpi @ x_mpi

    y_mpi.gather(x_glob)
    if rank == 0:
        assert (np.allclose(x_glob, z_glob))
