import argparse
import os
import sys

import numpy as np
import psutil
import pyamg
from mpi4py import MPI
from ngsolve import H1, InnerProduct, Preconditioner, ds, dx, grad, ngsglobals
from scipy.sparse.linalg.interface import LinearOperator

from bilform import BilForm
from fespace import KronFES
from linalg import PCG
from linform import LinForm
from linop import AsLinearOperator, CompositeLinOp
from mpi_kron import (BlockDiagMPI, CompositeMPI, MatKronIdentityMPI, SumMPI,
                      TridiagKronMatMPI)
from mpi_vector import KronVectorMPI
from problem import square
from wavelets import WaveletTransformOp


def mem():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1048576


def PyAMG(A):
    ml = pyamg.ruge_stuben_solver(A)

    def matvec(x):
        return ml.solve(x)

    return LinearOperator(matvec=matvec, shape=A.shape)


class HeatEquationMPI:
    def __init__(self, J_space=2, J_time=None, precond='multigrid', order=1):
        start_time = MPI.Wtime()
        if J_time is None:
            J_time = J_space

        mesh_space, bc_space, mesh_time, data, fn = square(J_space=J_space,
                                                           J_time=J_time)
        X = KronFES(H1(mesh_time, order=order),
                    H1(mesh_space, order=order, dirichlet=bc_space))
        self.N = len(X.time.fd)
        self.M = len(X.space.fd)
        self.mem_after_meshing = mem()

        # --- TIME ---
        self.J_time = J_time
        self.J_space = J_space
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
        self.W_t = WaveletTransformOp(self.J_time)

        # --- SPACE ---
        self.M_x = BilForm(X.space,
                           bilform_lambda=lambda u, v: u * v * dx).assemble()

        # Stiffness + multigrid preconditioner space.
        A_x = BilForm(
            X.space,
            bilform_lambda=lambda u, v: InnerProduct(grad(u), grad(v)) * dx)
        if precond != 'pyamg':
            Kinv_x_pc = Preconditioner(A_x.bf, precond)
        self.A_x = A_x.assemble()

        if precond != 'pyamg':
            self.Kinv_x = AsLinearOperator(Kinv_x_pc.mat, X.space.fd)
        else:
            self.Kinv_x = PyAMG(self.A_x)
        self.mem_after_space = mem()

        # --- Preconditioner on X ---
        self.C_j = []
        self.alpha = 0.5
        for j in range(self.J_time + 1):
            bf = BilForm(X.space,
                         bilform_lambda=lambda u, v:
                         (2**j * u * v + self.alpha * grad(u) * grad(v)) * dx)
            if precond != 'pyamg':
                C = Preconditioner(bf.bf, precond)
            bf.bf.Assemble()

            if precond != 'pyamg':
                self.C_j.append(AsLinearOperator(C.mat, X.space.fd))
            else:
                self.C_j.append(PyAMG(bf.assemble()))

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
        assert (len(data['g']) == 0)
        self.u0_t = LinForm(X.time, lambda v: v * ds('start')).assemble()
        self.u0_x = LinForm(X.space, lambda v: data['u0'] * v * dx).assemble()

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
    parser.add_argument('--precond',
                        default='multigrid',
                        help='type of preconditioner')
    parser.add_argument('--J_time',
                        type=int,
                        default=7,
                        help='number of time refines')
    parser.add_argument('--J_space',
                        type=int,
                        default=7,
                        help='number of space refines')

    args = parser.parse_args()
    precond = args.precond
    J_time = args.J_time
    J_space = args.J_space

    ngsglobals.msg_level = 0
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()
    if size > 2**J_time + 1:
        print('Too many MPI processors!')
        sys.exit('1')

    MPI.COMM_WORLD.Barrier()
    heat_eq_mpi = HeatEquationMPI(J_space=J_space,
                                  J_time=J_time,
                                  precond=precond)
    if rank == 0:
        print('\n\nCreating mesh with {} time refines and {} space refines.'.
              format(J_time, J_space))
        print('MPI tasks: ', size)
        print('Preconditioner: ', precond)
        print('N = {}. M = {}.'.format(heat_eq_mpi.N, heat_eq_mpi.M))
        print('Constructed bilinear forms in {} s.'.format(
            heat_eq_mpi.setup_time))
        print('\nMemory after meshing: {}mb.'.format(
            heat_eq_mpi.mem_after_meshing))
        print('Memory after space: {}mb.'.format(heat_eq_mpi.mem_after_space))
        print('Memory after precond: {}mb.'.format(
            heat_eq_mpi.mem_after_precond))
        print('Memory after construction: {}mb.'.format(mem()))
        print('')

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
