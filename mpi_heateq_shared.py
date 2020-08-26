import argparse
import pickle
import base64
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
from linop import AsLinearOperator, CompositeLinOp
from mpi_kron import (BlockDiagMPI, CompositeMPI, MatKronIdentityMPI, SumMPI,
                      TridiagKronMatMPI)
from mpi_vector import KronVectorMPI, DofDistributionMPI
from wavelets import (TransposedWaveletTransformKronIdentityMPI,
                      WaveletTransformKronIdentityMPI, WaveletTransformOp)


def mem():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1048576


class HeatEquationMPIShared:
    def __init__(self,
                 J_space=2,
                 J_time=None,
                 problem='square',
                 wavelettransform='original',
                 smoothsteps=3,
                 alpha=0.3,
                 vcycles=2):
        shared_comm = MPI.COMM_WORLD.Split_type(MPI.COMM_TYPE_SHARED)
        self.shared_comm = shared_comm

        start_time = MPI.Wtime()
        if J_time is None:
            J_time = J_space
        self.J_time = J_time
        self.J_space = J_space
        self.alpha = alpha

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
            from problem import problem_helper
            from linform import LinForm
            ngsglobals.msg_level = 0
            mesh_space, bc_space, mesh_time, data, fn = problem_helper(
                problem, J_space=J_space, J_time=J_time)
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
                self.Cinv_j[j] = 2**j * self.M_x + self.alpha * self.A_x

            # -- RHS --
            assert (len(data['g']) == 0)
            self.u0_t = LinForm(X.time, lambda v: v * ds('start')).assemble()
            self.u0_x = LinForm(X.space,
                                lambda v: data['u0'] * v * dx).assemble()
        self.mem_after_ngsolve = mem()

        # ---- Share variables with comm leader ----
        self.N, self.M = shared_comm.bcast((self.N, self.M))
        self.dofs_distr = DofDistributionMPI(MPI.COMM_WORLD, self.N, self.M)

        self.A_t = shared_sparse_matrix(self.A_t, shared_comm)
        self.L_t = shared_sparse_matrix(self.L_t, shared_comm)
        self.M_t = shared_sparse_matrix(self.M_t, shared_comm)
        self.G_t = shared_sparse_matrix(self.G_t, shared_comm)

        self.M_x = shared_sparse_matrix(self.M_x, shared_comm)
        self.A_x = shared_sparse_matrix(self.A_x, shared_comm)
        self.Cinv_j = [
            shared_sparse_matrix(mat, shared_comm) for mat in self.Cinv_j
        ]

        self.u0_t = shared_numpy_array(self.u0_t, shared_comm)
        self.u0_x = shared_numpy_array(self.u0_x, shared_comm)
        self.mem_after_shared_matrices = mem()

        # --- Wavelet transform ---
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

        self.P = BlockDiagMPI(self.dofs_distr,
                              [self.CAC_j[j] for j in self.W.levels])
        self.WT_S_W = CompositeMPI(self.dofs_distr, [self.WT, self.S, self.W])
        self.mem_after_mpi = mem()

        # -- RHS --
        self.rhs = KronVectorMPI(self.dofs_distr)
        self.rhs.X_loc[:] = np.kron(self.u0_t[self.rhs.t_begin:self.rhs.t_end],
                                    self.u0_x).reshape(-1, self.M)

        self.setup_time = MPI.Wtime() - start_time
        self.mem_after_mpi = mem()

    def print_time_per_apply(self):
        print('W:  {:.5f}\t{:.5f}'.format(*self.W.time_per_apply()))
        print('S:  {:.5f}\t{:.5f}'.format(*self.S.time_per_apply()))
        print('WT: {:.5f}\t{:.5f}'.format(*self.WT.time_per_apply()))
        print('P:  {:.5f}\t{:.5f}'.format(*self.P.time_per_apply()))
        print('')
        print('WTSW: {:.5f}\t{:.5f}'.format(*self.WT_S_W.time_per_apply()))


if __name__ == "__main__":
    import scipy.special as sc
    sc.seterr(all='raise')
    parser = argparse.ArgumentParser(
        description='Solve heatequation using MPI.')
    parser.add_argument('--problem',
                        default='square',
                        help='problem type (square, ns)')
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
    parser.add_argument('--wavelettransform',
                        default='original',
                        help='type of wavelettransform')
    parser.add_argument('--alpha', type=float, default=0.3, help='alpha')

    args = parser.parse_args()
    J_time = args.J_time
    J_space = args.J_space

    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()
    data = {'rank': rank, 'size': size}
    if size > 2**J_time + 1:
        print('Too many MPI processors!')
        sys.exit('1')

    heat_eq_mpi = HeatEquationMPIShared(J_space=J_space,
                                        J_time=J_time,
                                        problem=args.problem,
                                        smoothsteps=args.smoothsteps,
                                        vcycles=args.vcycles,
                                        alpha=args.alpha,
                                        wavelettransform=args.wavelettransform)
    if rank == 0:
        data['args'] = vars(args)
        data['N'] = heat_eq_mpi.N
        data['M'] = heat_eq_mpi.M
        print('\n\nCreating mesh with {} time refines and {} space refines.'.
              format(J_time, J_space))
        print('MPI tasks: ', size)
        print('Arguments:', args)
        print('N = {}. M = {}.'.format(heat_eq_mpi.N, heat_eq_mpi.M))
        print('Constructed bilinear forms in {} s.'.format(
            heat_eq_mpi.setup_time))
        print('Memory after ngsolve: {}mb.'.format(
            heat_eq_mpi.mem_after_ngsolve))
        print('Memory after shared mat: {}mb.'.format(
            heat_eq_mpi.mem_after_shared_matrices))
        print('Memory after precond: {}mb.'.format(
            heat_eq_mpi.mem_after_precond))
    data['mem_after_construction'] = mem()
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

    data['solve_time'] = MPI.Wtime() - solve_time
    data['mem_after_solve'] = mem()
    data['iters'] = iters
    for name, op in [('W', heat_eq_mpi.W), ('S', heat_eq_mpi.S),
                     ('WT', heat_eq_mpi.WT), ('P', heat_eq_mpi.P),
                     ('WT_S_W', heat_eq_mpi.WT_S_W)]:
        data[name] = {
            'time_applies': op.time_applies,
            'time_communication': op.time_communication,
            'num_applies': op.num_applies
        }

    MPI.COMM_WORLD.Barrier()
    if rank == 0:
        print('')
        print('Completed in {} PCG steps.'.format(iters))
        print('Total solve time: {}s.'.format(MPI.Wtime() - solve_time))
        heat_eq_mpi.print_time_per_apply()
        print('Memory after solve: {}mb.'.format(mem()))

    data = MPI.COMM_WORLD.gather(data, root=0)
    if rank == 0:
        print('\ndata: '.format(
            str(base64.b64encode(pickle.dumps(data)), 'ascii')))
