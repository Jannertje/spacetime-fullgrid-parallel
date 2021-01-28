import numpy as np
import scipy.sparse as sp
from ngsolve import BilinearForm, LinearForm, Preconditioner, dx, grad

from .linop import AsLinearOperator, BlockDiagLinOp, CompositeLinOp, KronLinOp
from .wavelets import WaveletTransformMat


class KronFES:
    """ Wrapper around ngsolve::FESpace. """
    def __init__(self, fes_time, fes_space):
        self.time = fes_time
        self.space = fes_space
        self.time.fd = [
            i for (i, free) in enumerate(fes_time.FreeDofs()) if free
        ]
        self.space.fd = [
            i for (i, free) in enumerate(fes_space.FreeDofs()) if free
        ]


class BilForm:
    """ Wrapper class around ngsolve for easier creation of bilforms. """
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
    """ Helper class that represents a kronecker ngsolve bilform. """
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


class LinForm:
    """ Wrapper around ngsolve::LinearForm. """
    def __init__(self, fes, linform_lambda):
        self.fes = fes
        self.lf = LinearForm(fes)
        self.lf += linform_lambda(self.fes.TestFunction())

    def assemble(self):
        self.lf.Assemble()
        self.vec = self.lf.vec.FV().NumPy()[self.fes.fd]
        return self.vec


class KronLF:
    """ Kronecker product of two LinForms. """
    def __init__(self, fes, time_lambda, space_lambda):
        self.time = LinForm(fes.time, time_lambda)
        self.space = LinForm(fes.space, space_lambda)

    def assemble(self):
        self.time.assemble()
        self.space.assemble()
        self.vec = np.kron(self.time.vec, self.space.vec)
        return self.vec
