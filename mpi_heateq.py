import argparse
import os
import sys

import numpy as np
from ngsolve import H1, InnerProduct, Preconditioner, ds, dx, grad, ngsglobals
from multigrid import MeshHierarchy, MultiGrid
from scipy.sparse.linalg.interface import LinearOperator

import psutil
from bilform import BilForm
from fespace import KronFES
from linalg import PCG
from linform import LinForm
from linop import AsLinearOperator, CompositeLinOp
from mpi4py import MPI
from mpi_kron import (BlockDiagMPI, CompositeMPI, MatKronIdentityMPI, SumMPI,
                      TridiagKronMatMPI)
from mpi_vector import KronVectorMPI, DofDistributionMPI
from problem import problem_helper
from wavelets import (TransposedWaveletTransformKronIdentityMPI,
                      WaveletTransformKronIdentityMPI, WaveletTransformOp)


def mem():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1048576


def PyAMG(A):
    import pyamg
    ml = pyamg.ruge_stuben_solver(A)

    def matvec(x):
        return ml.solve(x)

    return LinearOperator(matvec=matvec, shape=A.shape)


class HeatEquationMPI:
    def __init__(self,
                 J_space=2,
                 J_time=None,
                 precond='multigrid',
                 order=1,
                 problem='square',
                 wavelettransform='original',
                 smoothsteps=3,
                 alpha=0.3,
                 vcycles=2):
        precond_ngsolve = precond != 'mg' and precond != 'pyamg'
        start_time = MPI.Wtime()
        if J_time is None:
            J_time = J_space

        mesh_space, bc_space, mesh_time, data, fn = problem_helper(
            problem, J_space=J_space, J_time=J_time)

        X = KronFES(H1(mesh_time, order=order),
                    H1(mesh_space, order=order, dirichlet=bc_space))
        self.N = len(X.time.fd)
        self.M = len(X.space.fd)
        self.dofs_distr = DofDistributionMPI(MPI.COMM_WORLD, self.N, self.M)
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

        # --- SPACE ---
        self.M_x = BilForm(X.space,
                           bilform_lambda=lambda u, v: u * v * dx).assemble()

        # Stiffness + multigrid preconditioner space.
        A_x = BilForm(
            X.space,
            bilform_lambda=lambda u, v: InnerProduct(grad(u), grad(v)) * dx)
        if precond_ngsolve:
            Kinv_x_pc = Preconditioner(A_x.bf, precond)
        elif precond == 'mg':
            hierarchy = MeshHierarchy(X.space)
        self.A_x = A_x.assemble()

        if precond_ngsolve:
            self.Kinv_x = AsLinearOperator(Kinv_x_pc.mat, X.space.fd)
        elif precond == 'pyamg':
            self.Kinv_x = PyAMG(self.A_x)
        else:
            self.Kinv_x = MultiGrid(self.A_x,
                                    hierarchy,
                                    smoothsteps=smoothsteps,
                                    vcycles=vcycles)
        self.mem_after_space = mem()

        # --- Preconditioner on X ---
        self.C_j = []
        self.alpha = alpha
        for j in range(self.J_time + 1):
            bf = BilForm(X.space,
                         bilform_lambda=lambda u, v:
                         (2**j * u * v + self.alpha * grad(u) * grad(v)) * dx)
            if precond_ngsolve:
                C = Preconditioner(bf.bf, precond)
            bf.bf.Assemble()

            if precond_ngsolve:
                self.C_j.append(AsLinearOperator(C.mat, X.space.fd))
            elif precond == 'pyamg':
                self.C_j.append(PyAMG(bf.assemble()))
            else:
                self.C_j.append(
                    MultiGrid(bf.assemble(),
                              hierarchy,
                              smoothsteps=smoothsteps,
                              vcycles=vcycles))

        self.CAC_j = [
            CompositeLinOp([self.C_j[j], self.A_x, self.C_j[j]])
            for j in range(self.J_time + 1)
        ]
        self.mem_after_precond = mem()

        # -- MPI objects --
        self.A_MKM = TridiagKronMatMPI(
            self.dofs_distr, self.A_t,
            CompositeLinOp([self.M_x, self.Kinv_x, self.M_x]))
        self.L_MKA = TridiagKronMatMPI(
            self.dofs_distr, self.L_t,
            CompositeLinOp([self.M_x, self.Kinv_x, self.A_x]))
        self.LT_AKM = TridiagKronMatMPI(
            self.dofs_distr, self.L_t.T.tocsr(),
            CompositeLinOp([self.A_x, self.Kinv_x, self.M_x]))
        self.M_AKA = TridiagKronMatMPI(
            self.dofs_distr, self.M_t,
            CompositeLinOp([self.A_x, self.Kinv_x, self.A_x]))
        self.G_M = TridiagKronMatMPI(self.dofs_distr, self.G_t, self.M_x)
        self.S = SumMPI(
            self.dofs_distr,
            [self.A_MKM, self.L_MKA, self.LT_AKM, self.M_AKA, self.G_M])

        if wavelettransform == 'original':
            self.W_t = WaveletTransformOp(self.J_time)
            self.W = MatKronIdentityMPI(self.dofs_distr, self.W_t)
            self.WT = MatKronIdentityMPI(self.dofs_distr, self.W_t.H)
        elif wavelettransform == 'interleaved':
            self.W_t = WaveletTransformOp(self.J_time, interleaved=True)
            self.W = MatKronIdentityMPI(self.dofs_distr, self.W_t)
            self.WT = MatKronIdentityMPI(self.dofs_distr, self.W_t.H)
        elif wavelettransform == 'composite':
            self.W = WaveletTransformKronIdentityMPI(self.dofs_distr,
                                                     self.J_time)
            self.WT = TransposedWaveletTransformKronIdentityMPI(
                self.dofs_distr, self.J_time)
        self.P = BlockDiagMPI(self.dofs_distr,
                              [self.CAC_j[j] for j in self.W.levels])
        self.WT_S_W = CompositeMPI(self.dofs_distr, [self.WT, self.S, self.W])

        # -- RHS --
        assert (len(data['g']) == 0)
        self.u0_t = LinForm(X.time, lambda v: v * ds('start')).assemble()
        self.u0_x = LinForm(X.space, lambda v: data['u0'] * v * dx).assemble()

        self.rhs = KronVectorMPI(self.dofs_distr)
        self.rhs.X_loc[:] = np.kron(self.u0_t[self.rhs.t_begin:self.rhs.t_end],
                                    self.u0_x).reshape(-1, self.M)

        self.setup_time = MPI.Wtime() - start_time

    def print_time_per_apply(self):
        print('W: ', self.W.time_per_apply())
        print('S: ', self.S.time_per_apply())
        print('WT:', self.WT.time_per_apply())
        print('WTSWT:', self.WT_S_W.time_per_apply())
        print('P: ', self.P.time_per_apply())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Solve heatequation using MPI.')
    parser.add_argument('--problem',
                        default='square',
                        help='problem type (square, ns)')
    parser.add_argument('--precond',
                        default='multigrid',
                        help='type of preconditioner')
    parser.add_argument('--wavelettransform',
                        default='original',
                        help='type of preconditioner')
    parser.add_argument('--J_time',
                        type=int,
                        default=4,
                        help='number of time refines')
    parser.add_argument('--J_space',
                        type=int,
                        default=4,
                        help='number of space refines')
    parser.add_argument('--smoothsteps',
                        type=int,
                        default=3,
                        help='number of smoothing steps')
    parser.add_argument('--vcycles',
                        type=int,
                        default=2,
                        help='number of vcycles')
    parser.add_argument('--alpha', type=float, default=0.3, help='alpha')

    args = parser.parse_args()
    precond = args.precond
    wavelettransform = args.wavelettransform
    J_time = args.J_time
    J_space = args.J_space

    ngsglobals.msg_level = 0
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()
    if size > 2**J_time + 1:
        print('Too many MPI processors!')
        sys.exit('1')

    heat_eq_mpi = HeatEquationMPI(J_space=J_space,
                                  J_time=J_time,
                                  precond=precond,
                                  problem=args.problem,
                                  smoothsteps=args.smoothsteps,
                                  vcycles=args.vcycles,
                                  alpha=args.alpha,
                                  wavelettransform=wavelettransform)
    if rank == 0:
        print('\n\nCreating mesh with {} time refines and {} space refines.'.
              format(J_time, J_space))
        print('MPI tasks: ', size)
        print('Arguments:', args)
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

    MPI.COMM_WORLD.Barrier()
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
