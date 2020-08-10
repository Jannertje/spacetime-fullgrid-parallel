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


# Parallel-in-time only
def setup_local_communicators(X, Y):
    assert X.time.globalorder == 1 and Y.time.globalorder >= 1
    assert X.time.mesh.ne % mpi_world.size == 0
    assert mpi_world.size > 1

    n = mpi_world.rank
    N = mpi_world.size
    slice_size = int(X.time.mesh.ne / N)
    print("hier", n, N)
    mpi_timeslice = mpi_world.SubComm([(n - 1) % N, n, (n + 1) % N])
    print("daar")
    return mpi_timeslice


output = True
precond = 'multigrid'
order = 1

for N in [1, 2, 3, 4, 5, 6]:
    print("Building problem for N = {}".format(N))
    mesh_space, bc, mesh_time, data, fn = neumuller_smears(nrefines=N)

    print("Building fespaces")
    X = KronFES(H1(mesh_time, order=order),
                H1(mesh_space, order=order, dirichlet=bc))
    Y = KronFES(L2(mesh_time, order=order), X.space)

    if mpi_world.size > 1:
        mpi_timeslice = setup_local_communicators(X, Y)

    dt = dx
    print("Building bilforms")
    print("Building C. ", end='', flush=True)
    C_bf = KronBF(X, Y, lambda u, v: grad(u) * v * dt, lambda u, v: u * v * dx)
    print("done.")
    print("Building AXY. ", end='', flush=True)
    AXY_bf = KronBF(X, Y, lambda u, v: u * v * dt,
                    lambda u, v: InnerProduct(grad(u), grad(v)) * dx)
    print("done.")

    print("Building K. ", end='', flush=True)
    K_bf = KronBF(Y, Y, lambda u, v: u * v * dt,
                  lambda u, v: InnerProduct(grad(u), grad(v)) * dx)
    print("done.")
    A_bf = KronBF(X, X, lambda u, v: u * v * dt,
                  lambda u, v: InnerProduct(grad(u), grad(v)) * dx)
    A_t = BilForm(X.time, X.time, lambda u, v: grad(u) * grad(v) * dt)
    print("building Gamma_T. ", end='', flush=True)
    GT_bf = KronBF(X, X, lambda u, v: u * v * ds('end'),
                   lambda u, v: u * v * dx)
    print("done.")
    print("Assembling matrices. ", end='', flush=True)

    C = C_bf.assemble()
    AXY = AXY_bf.assemble()
    A = A_bf.assemble()
    CT = KronLinOp(C_bf.time.mat.T, C_bf.space.mat.T)
    A_t.assemble()
    GT = GT_bf.assemble()
    print("done.")

    print("building Kinv. ", end='', flush=True)
    Kinv_time_pc = Preconditioner(K_bf.time.bf, 'direct')
    Kinv_space_pc = Preconditioner(K_bf.space.bf, precond)
    K = K_bf.assemble()
    Kinv_time = AsLinearOperator(Kinv_time_pc.mat, Y.time.fd)
    Kinv_space = AsLinearOperator(Kinv_space_pc.mat, Y.space.fd)
    Kinv = KronLinOp(Kinv_time, Kinv_space)
    print("done.")
    print("building P. ", end='', flush=True)
    P = XPreconditioner(X, precond=precond)
    print("done")

    print("Building wavelet transform. ", end='', flush=True)
    W, WT = WaveletTransform(X)
    print("done.")

    print("Building S. ", end='', flush=True)
    Salt = sp.linalg.LinearOperator(
        GT.shape, matvec=lambda v: CT @ Kinv @ C @ v + A @ v + GT @ v)
    Yprime_linop = KronLinOp(
        A_t.mat, CompositeLinOp([GT_bf.space.mat, Kinv_space,
                                 GT_bf.space.mat]))
    S = sp.linalg.LinearOperator(
        GT.shape, matvec=lambda v: Yprime_linop @ v + A @ v + GT @ v)
    print("done.")
    start = time.process_time()
    v = np.random.random(GT.shape[0])
    out1 = Salt @ v
    print("NewMethod S apply speed: ", time.process_time() - start)
    start = time.process_time()
    out2 = S @ v
    print("Optimized S apply speed: ", time.process_time() - start)
    assert np.allclose(out1, out2)

    print(len(X.time.fd), len(X.space.fd), len(Y.time.fd))

    #lanczos = Lanczos(A=WT @ S @ W, P=K)
    #print(lanczos.cond())

    print("Building linforms. ", end='', flush=True)
    g_vec = np.zeros(K.shape[0])
    for g in data['g']:
        g_lf = KronLF(Y, lambda v: g[0] * v * dt, lambda v: g[1] * v * dx)
        g_lf.assemble()
        g_vec += g_lf.vec

    u0_lf = KronLF(X, lambda v: v * ds('start'), lambda v: data['u0'] * v * dx)
    u0_lf.assemble()

    CTKinvX = KronLinOp(CompositeLinOp([C_bf.time.mat.T, Kinv_time]),
                        sp.eye(GT_bf.space.mat.shape[0]))
    NKinvXid = KronLinOp(CompositeLinOp([AXY_bf.time.mat.T, Kinv_time]),
                         sp.eye(GT_bf.space.mat.shape[0]))
    f = CTKinvX @ g_vec + NKinvXid @ g_vec + u0_lf.vec
    print("done.")

    def cb(w, residual, k):
        if k % 10 == 0:
            v = W @ residual
            print('.', v.T @ v / (f.T @ f), flush=True)
        print('.', end='', flush=True)

    print("solving")
    w = PCG(WT @ S @ W, P, WT @ f, callback=cb)
    u = W @ w
    gminBu = g_vec - AXY @ u - C @ u
    print("done. X-norm error: ", gminBu.T @ (Kinv @ gminBu))
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
