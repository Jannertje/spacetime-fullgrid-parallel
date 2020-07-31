from ngsolve import LinearForm
import numpy as np


class KronLF:
    def __init__(self, fes, time_lambda, space_lambda):
        self.fes = fes
        self.time = LinearForm(fes.time)
        self.time += time_lambda(self.fes.time.TestFunction())

        self.space = LinearForm(fes.space)
        self.space += space_lambda(self.fes.space.TestFunction())

    def assemble(self):
        self.time.Assemble()
        self.space.Assemble()
        self.vec = np.kron(self.time.vec.FV().NumPy()[self.fes.time.fd],
                           self.space.vec.FV().NumPy()[self.fes.space.fd])
