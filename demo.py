import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg
from ngsolve import *

from bilform import *
from fespace import *
from lanczos import *
from linalg import *
from linform import *
from linop import *
from mesh import *
from problem import *

ngsglobals.msg_level = 0


class HeatEquation:
    def __init__(self,
                 mesh_space,
                 bc,
                 mesh_time,
                 data,
                 precond='multigrid',
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
        B1_bf.assemble()
        B2_bf.assemble()
        self.B = KronLinOp(B1_bf.time.mat + B2_bf.time.mat,
                           B1_bf.space.mat + B2_bf.space.mat)
        self.BT = KronLinOp(B1_bf.time.mat.T + B2_bf.time.mat.T,
                            B1_bf.space.mat.T + B2_bf.space.mat.T)
        self.G = G_bf.assemble()

        # Preconditioners.
        Kinv_time_pc = Preconditioner(A_bf.time.bf, 'direct')
        Kinv_space_pc = Preconditioner(A_bf.space.bf, precond)
        A_bf.assemble()
        Kinv_time = AsLinearOperator(Kinv_time_pc.mat, Y.time.fd)
        Kinv_space = AsLinearOperator(Kinv_space_pc.mat, Y.space.fd)
        self.K = KronLinOp(Kinv_time, Kinv_space)
        self.P = XPreconditioner(X, precond=precond)

        self.W, self.WT = WaveletTransform(X)

        # Schur-complement operator.
        self.S = sp.linalg.LinearOperator(
            self.G.shape,
            matvec=lambda v: self.BT @ self.K @ self.B @ v + self.G @ v)

        # Calculate rhs.
        g_vec = np.zeros(self.K.shape[0])
        for g in data['g']:
            g_lf = KronLF(Y, lambda v: g[0] * v * dt, lambda v: g[1] * v * dx)
            g_lf.assemble()
            g_vec += g_lf.vec
        u0_lf = KronLF(X, lambda v: v * ds('start'),
                       lambda v: data['u0'] * v * dx)
        u0_lf.assemble()

        self.f = self.BT @ self.K @ g_vec + u0_lf.vec


if __name__ == '__main__':
    output = True
    precond = 'direct'
    order = 1

    for N in [1, 2, 3, 4, 5, 6]:
        print("Building problem for N = {}".format(N))
        mesh_space, bc, mesh_time, data, fn = square(N)
        heat_eq = HeatEquation(mesh_space, bc, mesh_time, data, precond, order)

        print('cond(P @ WT @ S @ W)',
              Lanczos(heat_eq.WT @ heat_eq.S @ heat_eq.W, heat_eq.P))

        def cb(w, residual, k):
            if k % 10 == 0:
                v = W @ residual
                print('.', v.T @ v / (f.T @ f), flush=True)
            print('.', end='', flush=True)

        print("solving")
        w, iters = PCG(heat_eq.WT @ heat_eq.S @ heat_eq.W,
                       heat_eq.P,
                       WT @ f,
                       callback=cb)
        u = heat_eq.W @ w
        gminBu = g_vec - AXY @ u - C @ u
        print("done in ", iters, " PCG steps. X-norm error: ",
              gminBu.T @ (Kinv @ gminBu))
        if output:
            print(np.max(u))
            u_dekron = u.reshape(len(X.time.fd), len(X.space.fd))
            gf_space = GridFunction(X.space)
            vtk = VTKOutput(ma=mesh_space,
                            coefs=[gf_space],
                            names=['u'],
                            filename='output/%s_%d_now' % (fn, N),
                            subdivision=order - 1)
            for t in range(mesh_time.nv):
                gf_space.vec.FV().NumPy()[X.space.fd] = u_dekron[t, :]
                vtk.Do()
