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
        self.mat.eliminate_zeros()
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
        return KronLinOp(self.time.mat, self.space.mat)

    def transpose(self):
        return KronLinOp(self.time.mat.T, self.space.mat.T)
