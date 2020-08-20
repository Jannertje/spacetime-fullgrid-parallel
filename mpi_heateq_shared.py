import argparse
import os
from shared_mem import shared_numpy_array, shared_sparse_matrix
import sys

import numpy as np
import psutil
import pyamg
from mpi4py import MPI
from multigrid import MeshHierarchy, MultiGrid
from scipy.sparse.linalg.interface import LinearOperator

from linalg import PCG
from linform import LinForm
from linop import AsLinearOperator, CompositeLinOp
from mpi_kron import (BlockDiagMPI, CompositeMPI, MatKronIdentityMPI, SumMPI,
                      TridiagKronMatMPI)
from mpi_vector import KronVectorMPI
from wavelets import WaveletTransformOp


def mem():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1048576


class HeatEquationMPIShared:
    def __init__(self, J_space=2, J_time=None, smoothsteps=3, vcycles=2):
        shared_comm = MPI.COMM_WORLD.Split_type(MPI.COMM_TYPE_SHARED)
        self.shared_comm = shared_comm

        start_time = MPI.Wtime()
        if J_time is None:
            J_time = J_space
        self.J_time = J_time
        self.J_space = J_space
        self.alpha = 0.5

        # Set variables to zero.
        self.N = self.M = None
        self.A_t = self.L_t = self.M_t = self.G_t = None
        self.M_x = self.A_x = None
        self.Cinv_j = [None for _ in range(self.J_time + 1)]
        self.u0_t = self.u0_x = None
        fes_x = None

        if shared_comm.rank == 0:
            from ngsolve import H1, InnerProduct, Preconditioner, ds, dx, grad, ngsglobals
            from bilform import BilForm
            from fespace import KronFES
            from problem import square
            ngsglobals.msg_level = 0
            mesh_space, bc_space, mesh_time, data, fn = square(J_space=J_space,
                                                               J_time=J_time)
            X = KronFES(H1(mesh_time, order=1),
                        H1(mesh_space, order=1, dirichlet=bc_space))
            self.N = len(X.time.fd)
            self.M = len(X.space.fd)
            self.mem_after_meshing = mem()

            # --- TIME ---
            self.A_t = BilForm(
                X.time,
                bilform_lambda=lambda u, v: grad(u) * grad(v) * dx).assemble()
            self.L_t = BilForm(
                X.time,
                bilform_lambda=lambda u, v: u * grad(v) * dx).assemble()
            self.M_t = BilForm(
                X.time, bilform_lambda=lambda u, v: u * v * dx).assemble()
            self.G_t = BilForm(
                X.time,
                bilform_lambda=lambda u, v: u * v * ds('start')).assemble()

            # --- SPACE ---
            fes_x = X.space
            self.M_x = BilForm(
                X.space, bilform_lambda=lambda u, v: u * v * dx).assemble()
            self.A_x = BilForm(X.space,
                               bilform_lambda=lambda u, v: InnerProduct(
                                   grad(u), grad(v)) * dx).assemble()
            for j in range(self.J_time + 1):
                self.Cinv_j[j] = 2**j * self.M_x + self.A_x

            # -- RHS --
            assert (len(data['g']) == 0)
            self.u0_t = LinForm(X.time, lambda v: v * ds('start')).assemble()
            self.u0_x = LinForm(X.space,
                                lambda v: data['u0'] * v * dx).assemble()
            self.mem_after_ngsolve = mem()

        # Set variables to according to the leader.
        self.N, self.M = shared_comm.bcast((self.N, self.M))

        self.A_t = shared_sparse_matrix(self.A_t, shared_comm)
        self.L_t = shared_sparse_matrix(self.L_t, shared_comm)
        self.M_t = shared_sparse_matrix(self.M_t, shared_comm)
        self.G_t = shared_sparse_matrix(self.G_t, shared_comm)
        self.W_t = WaveletTransformOp(self.J_time)

        self.M_x = shared_sparse_matrix(self.M_x, shared_comm)
        self.A_x = shared_sparse_matrix(self.A_x, shared_comm)
        self.Cinv_j = [
            shared_sparse_matrix(mat, shared_comm) for mat in self.Cinv_j
        ]

        self.u0_t = shared_numpy_array(self.u0_t, shared_comm)
        self.u0_x = shared_numpy_array(self.u0_x, shared_comm)
        self.mem_after_shared_matrices = mem()

        # ---- Preconditioners in space ----
        hierarchy = MeshHierarchy(fes_x, shared_comm)
        self.Kinv_x = MultiGrid(self.A_x,
                                hierarchy,
                                smoothsteps=smoothsteps,
                                vcycles=vcycles)
        self.C_j = [
            MultiGrid(mat, hierarchy, smoothsteps=smoothsteps, vcycles=vcycles)
            for mat in self.Cinv_j
        ]
        self.CAC_j = [
            CompositeLinOp([self.C_j[j], self.A_x, self.C_j[j]])
            for j in range(self.J_time + 1)
        ]
        self.mem_after_precond = mem()

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
        self.rhs = KronVectorMPI(MPI.COMM_WORLD, self.N, self.M)
        self.rhs.X_loc[:] = np.kron(self.u0_t[self.rhs.t_begin:self.rhs.t_end],
                                    self.u0_x).reshape(-1, self.M)

        self.setup_time = MPI.Wtime() - start_time

    def print_time_per_apply(self):
        print('W: ', self.W.time_per_apply())
        print('S: ', self.S.time_per_apply())
        print('WT:', self.WT.time_per_apply())
        print('P: ', self.P.time_per_apply())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Solve heatequation using MPI.')
    parser.add_argument('--J_time',
                        type=int,
                        default=7,
                        help='number of time refines')
    parser.add_argument('--J_space',
                        type=int,
                        default=7,
                        help='number of space refines')
    parser.add_argument('--smoothsteps',
                        type=int,
                        default=3,
                        help='number of smoothing steps')
    parser.add_argument('--vcycles',
                        type=int,
                        default=2,
                        help='number of vcycles')

    args = parser.parse_args()
    J_time = args.J_time
    J_space = args.J_space

    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()
    if size > 2**J_time + 1:
        print('Too many MPI processors!')
        sys.exit('1')

    MPI.COMM_WORLD.Barrier()
    heat_eq_mpi = HeatEquationMPIShared(J_space=J_space,
                                        J_time=J_time,
                                        smoothsteps=args.smoothsteps,
                                        vcycles=args.vcycles)
    if rank == 0:
        print('\n\nCreating mesh with {} time refines and {} space refines.'.
              format(J_time, J_space))
        print('MPI tasks: ', size)
        print('Smoothsteps:', args.smoothsteps, '. Vcycles:', args.vcycles)
        print('N = {}. M = {}.'.format(heat_eq_mpi.N, heat_eq_mpi.M))
        print('Constructed bilinear forms in {} s.'.format(
            heat_eq_mpi.setup_time))
        print('Memory after ngsolve: {}mb.'.format(
            heat_eq_mpi.mem_after_ngsolve))
        print('Memory after shared mat: {}mb.'.format(
            heat_eq_mpi.mem_after_shared_matrices))
        print('Memory after precond: {}mb.'.format(
            heat_eq_mpi.mem_after_precond))
    if rank < 2:
        print('Memory {} after construction: {}mb.'.format(rank, mem()))

    # Solve.
    def cb(w, residual, k):
        if rank == 0:
            print('.', end='', flush=True)

    solve_time = MPI.Wtime()
    u_mpi_P, iters = PCG(heat_eq_mpi.WT_S_W,
                         heat_eq_mpi.P,
                         heat_eq_mpi.rhs,
                         callback=cb)

    if rank == 0:
        print('')
        print('Completed in {} PCG steps.'.format(iters))
        print('Total solve time: {}s.'.format(MPI.Wtime() - solve_time))
        heat_eq_mpi.print_time_per_apply()
        print('Memory after solve: {}mb.'.format(mem()))
