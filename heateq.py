import argparse
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg
from ngsolve import (H1, L2, InnerProduct, Preconditioner, ds, dx, grad,
                     ngsglobals)

from source.linalg import PCG
from source.linop import (AsLinearOperator, BlockDiagLinOp, CompositeLinOp,
                          KronLinOp)
from source.mpi_kron import as_matrix
from source.ngsolve_helper import BilForm, KronBF, KronFES, KronLF, LinForm
from source.problem import problem_helper
from source.wavelets import WaveletTransformOp

ngsglobals.msg_level = 0


class HeatEquation:
    """ Implementation of Andreev's method for tensor-product trial spaces. """
    def __init__(self,
                 J_space=2,
                 J_time=None,
                 problem='square',
                 precond='multigrid',
                 alpha=0.3,
                 order=1):
        if J_time is None:
            J_time = J_space
        mesh_space, bc, mesh_time, data, fn = problem_helper(problem,
                                                             J_space=J_space,
                                                             J_time=J_time)

        # Building fespaces X^\delta and Y^\delta
        X = KronFES(H1(mesh_time, order=order),
                    H1(mesh_space, order=order, dirichlet=bc))
        Y = KronFES(L2(mesh_time, order=order), X.space)
        self.N = len(X.time.fd)
        self.M = len(X.space.fd)

        # Building the ngsolve spacetime-bilforms.
        dt = dx
        A_bf = KronBF(Y, Y, lambda u, v: u * v * dt,
                      lambda u, v: InnerProduct(grad(u), grad(v)) * dx)
        B1_bf = KronBF(X, Y, lambda u, v: grad(u) * v * dt,
                       lambda u, v: u * v * dx)
        B2_bf = KronBF(X, Y, lambda u, v: u * v * dt,
                       lambda u, v: InnerProduct(grad(u), grad(v)) * dx)
        G_bf = KronBF(X, X, lambda u, v: u * v * ds('start'),
                      lambda u, v: u * v * dx)

        # Create the kron linops.
        self.B = B1_bf.assemble() + B2_bf.assemble()
        self.BT = B1_bf.transpose() + B2_bf.transpose()
        self.G = G_bf.assemble()

        # Preconditioner on Y.
        Kinv_time_pc = Preconditioner(A_bf.time.bf, 'direct')
        Kinv_space_pc = Preconditioner(A_bf.space.bf, precond)
        A_bf.assemble()
        Kinv_time = AsLinearOperator(Kinv_time_pc.mat, Y.time.fd)
        Kinv_space = AsLinearOperator(Kinv_space_pc.mat, Y.space.fd)
        self.K = KronLinOp(Kinv_time, Kinv_space)

        # --- Wavelet transform ---
        W_t = WaveletTransformOp(J_time)
        self.W = KronLinOp(W_t, sp.eye(len(X.space.fd)))
        self.WT = KronLinOp(W_t.T, sp.eye(len(X.space.fd)))

        # --- Preconditioner on X ---
        self.C_j = []
        self.alpha = alpha
        for j in range(J_time + 1):
            bf = BilForm(X.space,
                         bilform_lambda=lambda u, v:
                         (2**j * u * v + self.alpha * grad(u) * grad(v)) * dx)
            C = Preconditioner(bf.bf, precond)
            bf.bf.Assemble()
            self.C_j.append(AsLinearOperator(C.mat, X.space.fd))

        self.CAC_j = [
            CompositeLinOp([self.C_j[j], A_bf.space.mat, self.C_j[j]])
            for j in range(J_time + 1)
        ]
        self.P = BlockDiagLinOp([self.CAC_j[j] for j in W_t.levels])

        # Schur-complement operator.
        self.S = sp.linalg.LinearOperator(
            self.G.shape,
            matvec=lambda v: self.BT @ self.K @ self.B @ v + self.G @ v)
        self.WT_S_W = self.WT @ self.S @ self.W

        # Calculate rhs.
        self.g_vec = np.zeros(self.K.shape[0])
        for g in data['g']:
            g_lf = KronLF(Y, lambda v: g[0] * v * dt, lambda v: g[1] * v * dx)
            g_lf.assemble()
            self.g_vec += g_lf.vec
        u0_lf = KronLF(X, lambda v: v * ds('start'),
                       lambda v: data['u0'] * v * dx)
        u0_lf.assemble()

        self.f = self.BT @ self.K @ self.g_vec + u0_lf.vec


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Solve heatequation using ngsolve.')
    parser.add_argument('--problem',
                        default='square',
                        help='problem type (square, ns)')
    parser.add_argument('--J_time',
                        type=int,
                        default=5,
                        help='number of time refines')
    parser.add_argument('--J_space',
                        type=int,
                        default=6,
                        help='number of space refines')
    parser.add_argument(
        '--precond',
        default="multigrid",
        help='type of ngsolve preconditioner, e.g. direct or multigrid.')
    parser.add_argument('--alpha',
                        type=float,
                        default=0.3,
                        help='Alpha value used in the preconditioner for X.')
    args = parser.parse_args()
    order = 1  # Higher order requires a different wavelet transform.

    print('Arguments: {}'.format(args))
    print(
        '\n\nCreating HeatEquation with {} time refines and {} space refines.'.
        format(args.J_time, args.J_space))
    heat_eq = HeatEquation(J_time=args.J_time,
                           J_space=args.J_space,
                           problem=args.problem,
                           precond=args.precond,
                           alpha=args.alpha)
    print('Size of time mesh: {} dofs. Size of space mesh: {} dofs'.format(
        heat_eq.N, heat_eq.M))

    def cb(w, residual, k):
        print('.', end='', flush=True)

    print("Solving: ", end='')
    w, iters = PCG(heat_eq.WT_S_W,
                   heat_eq.P,
                   heat_eq.WT @ heat_eq.f,
                   callback=cb)
    u = heat_eq.W @ w
    res = heat_eq.f - heat_eq.S @ heat_eq.f
    error_alg = res @ (heat_eq.P @ res)

    gminBu = heat_eq.g_vec - heat_eq.B @ u
    error_Yprime = gminBu @ (heat_eq.K @ gminBu)
    print(
        "Done in {}  PCG steps. X-norm algebraic error: {}. Error in Yprime: {}\n"
        .format(iters, error_alg, error_Yprime))
