import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg.interface import LinearOperator

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.size


class WaveletKronIdentity(LinearOperator):
    def __init__(self, comm, J, space_size):
        self.comm = comm
        self.J = J
        self.N = 2**self.J + 1
        self.M = space_size
        super().__init__(dtype=np.float64,
                         shape=(self.N * self.M, self.N * self.M))
        self.T_ = self._q(0)
        for j in range(1, J + 1):
            self.T_ = sp.bmat([[self._p(j) @ self.T_, self._q(j)]])

    def _p(self, j):
        mat = sp.csr_matrix((2**(j - 1) + 1, 2**j + 1))
        mat[:, 0::2] = sp.eye(2**(j - 1) + 1)
        mat[:, 1::2] = sp.diags([0.5, 0.5], [0, -1],
                                shape=(2**(j - 1) + 1, 2**(j - 1)))
        return mat.T

    def _q(self, j):
        if j == 0: return sp.eye(2)
        mat = sp.csr_matrix((2**(j - 1), 2**j + 1))
        mat[:, 0::2] = sp.diags([-0.5, -0.5], [0, 1],
                                shape=(2**(j - 1), 2**(j - 1) + 1))
        mat[:, 1::2] = sp.eye(2**(j - 1))
        mat[0, 0] = -1
        mat[-1, -1] = -1
        mat *= 2**(j / 2)
        return mat.T

    def _matvec(self, x):
        Y = x.reshape(self.N, self.M)
        for j in range(1, self.J + 1):
            interleaved = np.zeros((2**j + 1, 2**j + 1))
            interleaved[:, ::2] = self._p(j).toarray()
            interleaved[:, 1::2] = self._q(j).toarray()
            plt.spy(interleaved)
            plt.show()
            stride = 2**(self.J - j)
            Y[0::2 * stride], Y[1::2 * stride] = self._p(
                j).T @ Y[::stride], self._q(j).T @ Y[::stride]
            #old = 2**j + 1
            #new = 2**(j - 1) + 1
            #Y[:old] = self._p(j) @ Y[:new] + self._q(j) @ Y[new:old]
        return Y.reshape(-1)

    def _rmatvec(self, x):
        Y = x.reshape(self.N, self.M)
        for j in reversed(range(1, self.J + 1)):
            stride = 2**(self.J - j)
            Y[0::2 * stride], Y[1::2 * stride] = self._p(
                j).T @ Y[::stride], self._q(j).T @ Y[::stride]
            #old = 2**j + 1
            #new = 2**(j - 1) + 1
            #Y[:new], Y[new:old] = self._p(j).T @ Y[:old], self._q(
            #    j).T @ Y[:old]
        return Y.reshape(-1)


W = WaveletKronIdentity(comm, J=4, space_size=1)
#assert np.allclose(W.T_.toarray().T, W.T @ np.eye(2**4 + 1))
plt.spy(W @ np.eye(2**4 + 1))
plt.show()
assert np.allclose(W.T_.toarray(), W @ np.eye(2**4 + 1))
