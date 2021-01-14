import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg
from ngsolve import H1, InnerProduct, Preconditioner, ds, dx, grad, ngsglobals, L2

from mpi_kron import as_matrix
from wavelets import WaveletTransformOp
from bilform import KronBF, BilForm
from linalg import PCG
from linop import AsLinearOperator, KronLinOp, CompositeLinOp, BlockDiagLinOp
from linform import LinForm, KronLF
from fespace import KronFES
from problem import square, cube

ngsglobals.msg_level = 0


class HeatEquation:
    def __init__(self,
                 mesh_space,
                 bc,
                 mesh_time,
                 data,
                 fn,
                 precond='multigrid',
                 alpha=0.3,
                 order=1):
        # Building fespaces X^\delta and Y^\delta
        X = KronFES(H1(mesh_time, order=order),
                    H1(mesh_space, order=order, dirichlet=bc))
        Y = KronFES(L2(mesh_time, order=order), X.space)

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
        J_time = int(np.log(X.time.mesh.ne) / np.log(2))
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
    output = True
    precond = 'direct'
    order = 1  # Higher order requires a different wavelet transform.

    for N in [1, 2, 3, 4, 5, 6]:
        print("Building problem for N = {}".format(N))
        heat_eq = HeatEquation(*square(N), precond, order)

        def cb(w, residual, k):
            print('.', end='', flush=True)

        print("solving")
        w, iters = PCG(heat_eq.WT @ heat_eq.S @ heat_eq.W,
                       heat_eq.P,
                       heat_eq.WT @ heat_eq.f,
                       callback=cb)
        u = heat_eq.W @ w
        res = heat_eq.f - heat_eq.S @ heat_eq.f
        error_alg = res @ (heat_eq.P @ res)

        gminBu = heat_eq.g_vec - heat_eq.B @ u
        error_Yprime = gminBu @ (heat_eq.K @ gminBu)
        print(
            "done in {}  PCG steps. X-norm algebraic error: {}. Error in Yprime: {}\n"
            .format(iters, error_alg, error_Yprime))
