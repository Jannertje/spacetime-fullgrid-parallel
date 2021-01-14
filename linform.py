import numpy as np
from ngsolve import LinearForm


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
