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
from lanczos import Lanczos
from linalg import PCG
from linform import LinForm
from linop import AsLinearOperator, CompositeLinOp
from mpi_kron import (BlockDiagMPI, CompositeMPI, IdentityKronMatMPI,
                      IdentityMPI, LinearOperatorMPI, MatKronIdentityMPI,
                      SumMPI, TridiagKronIdentityMPI, TridiagKronMatMPI,
                      as_matrix)
from mpi_vector import KronVectorMPI
from problem import cube, square
from wavelets import WaveletTransformOp


class HeatEquationMPI:
    def __init__(self, refines=2, precond='multigrid', order=1):
        mesh_space, bc_space, mesh_time, data, fn = square(nrefines=refines)
        X = KronFES(H1(mesh_time, order=order),
                    H1(mesh_space, order=order, dirichlet=bc_space))
        self.N = len(X.time.fd)
        self.M = len(X.space.fd)

        # --- TIME ---
        self.J = refines
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
        self.W_t = WaveletTransformOp(self.J)

        # --- SPACE ---
        self.M_x = BilForm(X.space,
                           bilform_lambda=lambda u, v: u * v * dx).assemble()

        # Stiffness + multigrid preconditioner space.
        A_x = BilForm(
            X.space,
            bilform_lambda=lambda u, v: InnerProduct(grad(u), grad(v)) * dx)
        Kinv_x_pc = Preconditioner(A_x.bf, precond)
        self.A_x = A_x.assemble()
        self.Kinv_x = AsLinearOperator(Kinv_x_pc.mat, X.space.fd)

        # --- Preconditioner on X ---
        self.C_j = []
        self.alpha = 0.5
        for j in range(self.J + 1):
            bf = BilForm(
                X.space,
                bilform_lambda=lambda u, v:
                (2**j * u * v + self.alpha * grad(u) * grad(v)) * dx).bf
            C = Preconditioner(bf, precond)
            bf.Assemble()
            self.C_j.append(AsLinearOperator(C.mat, X.space.fd))

        self.CAC_j = [
            CompositeLinOp([self.C_j[j], self.A_x, self.C_j[j]])
            for j in range(self.J + 1)
        ]

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
        self.S = SumMPI(
            [self.A_MKM, self.L_MKA, self.LT_AKM, self.M_AKA, self.G_M])

        self.P = BlockDiagMPI([self.CAC_j[j] for j in self.W_t.levels])

        self.W = MatKronIdentityMPI(self.W_t, self.M)
        self.WT = MatKronIdentityMPI(self.W_t.H, self.M)

        self.WT_S_W = CompositeMPI([self.WT, self.S, self.W])

        # -- RHS --
        assert (len(data['g']) == 0)
        self.u0_t = LinForm(X.time, lambda v: v * ds('start')).assemble()
        self.u0_x = LinForm(X.space, lambda v: data['u0'] * v * dx).assemble()

        self.rhs = KronVectorMPI(MPI.COMM_WORLD, self.N, self.M)
        self.rhs.X_loc = np.kron(self.u0_t[self.rhs.t_begin:self.rhs.t_end],
                                 self.u0_x).reshape(-1, self.M)


if __name__ == "__main__":
    refines = 4
    heat_eq_mpi = HeatEquationMPI(refines)
    N = heat_eq_mpi.N
    M = heat_eq_mpi.M
    comm = MPI.COMM_WORLD

    # Create random MPI vector.
    w_mpi = KronVectorMPI(comm, N, M)
    w_mpi.X_loc = np.random.rand(w_mpi.X_loc.shape[0], M)
    print(w_mpi.X_loc)

    # Perform Lanczos.
    lanczos = Lanczos(heat_eq_mpi.WT_S_W, heat_eq_mpi.P, w=w_mpi)

    if w_mpi.rank == 0:
        print(N * M, lanczos)

        # Compare to demo
        _, _, WT, S, W, _, P, _, _, _, _ = demo(*square(refines))
        #print('cond(P @ WT @ S @ W)', Lanczos(WT @ S @ W, P))
