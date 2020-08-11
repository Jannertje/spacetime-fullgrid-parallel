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
from linalg import PCG
from linform import LinForm
from linop import AsLinearOperator, CompositeLinOp
from mpi_kron import (IdentityKronMatMPI, IdentityMPI, LinearOperatorMPI,
                      TridiagKronIdentityMPI, TridiagKronMatMPI, as_matrix)
from mpi_vector import KronVectorMPI
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

        # -- RHS --
        assert (len(data['g']) == 0)
        self.u0_t = LinForm(X.time, lambda v: v * ds('start')).assemble()
        self.u0_x = LinForm(X.space, lambda v: data['u0'] * v * dx).assemble()

        self.rhs = KronVectorMPI(MPI.COMM_WORLD, self.N, self.M)
        self.rhs.X_loc = np.kron(self.u0_t[self.rhs.t_begin:self.rhs.t_end],
                                 self.u0_x).reshape(-1, self.M)

    def matvec(self, vec_in, vec_out):
        assert (vec_in is not vec_out)
        Y_loc = np.zeros(vec_out.X_loc.shape)

        for linop in self.linops:
            linop.matvec(vec_in, vec_out)
            Y_loc += vec_out.X_loc

        vec_out.X_loc = Y_loc
        return vec_out
