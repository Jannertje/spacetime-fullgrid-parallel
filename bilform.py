import numpy as np
import scipy.sparse as sp
from ngsolve import BilinearForm, Preconditioner, dx, grad

from linop import AsLinearOperator, BlockDiagLinOp, CompositeLinOp, KronLinOp
from wavelets import WaveletTransformMat


class BilForm:
    def __init__(self, fes_in, fes_out=None, bilform_lambda=None):
        self.fes_in = fes_in
        self.fes_out = fes_out if fes_out else fes_in
        if self.fes_in is self.fes_out:
            self.bf = BilinearForm(self.fes_in,
                                   symmetric=False,
                                   check_unused=False)
        else:
            self.bf = BilinearForm(self.fes_in,
                                   self.fes_out,
                                   symmetric=False,
                                   check_unused=False)
        self.bf += bilform_lambda(self.fes_in.TrialFunction(),
                                  self.fes_out.TestFunction())

    def assemble(self):
        self.bf.Assemble()
        mat = sp.csr_matrix(self.bf.mat.CSR())
        self.mat = mat[self.fes_out.FreeDofs(), :].tocsc()[:,
                                                           self.fes_in.
                                                           FreeDofs()].tocsr()
        return self.mat


class KronBF:
    def __init__(self,
                 fes_in,
                 fes_out=None,
                 bilform_time_lambda=None,
                 bilform_space_lambda=None):
        self.fes_in = fes_in
        self.fes_out = fes_out if fes_out else fes_in
        self.time = BilForm(self.fes_in.time, self.fes_out.time,
                            bilform_time_lambda)
        self.space = BilForm(self.fes_in.space, self.fes_out.space,
                             bilform_space_lambda)

    def assemble(self):
        self.time.assemble()
        self.space.assemble()
        self.linop = KronLinOp(self.time.mat, self.space.mat)
        return self.linop


def WaveletTransform(fes):
    J = int(np.log(fes.time.mesh.ne) / np.log(2))
    T = WaveletTransformMat(J)
    if fes.time.globalorder > 1:
        T = sp.block_diag([T, sp.eye(len(fes.time.fd) - fes.time.mesh.nv)
                           ]).tocsr()

    return (KronLinOp(T, sp.eye(len(fes.space.fd))),
            KronLinOp(T.T, sp.eye(len(fes.space.fd))))


def XPreconditioner(fes, precond='multigrid', alpha=0.5):
    J = int(np.log(fes.time.mesh.ne) / np.log(2))
    K_mus = []
    u, v = fes.space.TnT()
    for j in range(J + 1):
        bf = BilinearForm(fes.space)
        bf += (2**j * u * v + alpha * grad(u) * grad(v)) * dx
        K = Preconditioner(bf, precond)
        bf.Assemble()
        K_mus.append(AsLinearOperator(K.mat, fes.space.fd))
    multiples = [2, 1] + [2**(j - 1) for j in range(2, J + 1)]
    levels = []
    for j, m in enumerate(multiples):
        levels = levels + [j for _ in range(m)]

    A_bf = BilinearForm(fes.space)
    A_bf += grad(u) * grad(v) * dx
    A_bf.Assemble()
    A_mu = sp.csr_matrix(A_bf.mat.CSR())
    A_mu = A_mu[fes.space.fd, :]
    A_mu = A_mu[:, fes.space.fd]

    return BlockDiagLinOp(
        [CompositeLinOp([K_mus[j], A_mu, K_mus[j]]) for j in levels])
