import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg.interface import LinearOperator

import wavelets
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.size


class LevelWaveletTransformOp(wavelets.WaveletTransformOp):
    def __init__(self, J):
        super().__init__(J=J, interleaved=True)
        self.split = []
        for j in range(J + 1):
            self.split.append(
                LevelWaveletTransformOp._split(J, j, self.p[j], self.q[j]))
        plt.imshow(
            np.log10(
                np.abs(np.hstack([split.toarray()
                                  for split in self.split[1:]]))))
        plt.show()

    def _split(J, j, p, q):
        if j == 0: return sp.csr_matrix((2**J + 1, 2**J + 1))
        I = np.eye(2**J + 1)
        mat = np.eye(2**J + 1)
        S = 2**(J - j)
        mat[::S] = p @ I[::S][::2] + q @ I[::S][1::2]
        mat -= sp.eye(2**J + 1)
        return sp.csr_matrix(mat)

    def _matmat(self, x):
        y = x.copy()
        for j in range(1, self.J + 1):
            y += self.split[j] @ y
        return y

    def _rmatmat(self, x):
        y = x.copy()
        for j in reversed(range(1, self.J + 1)):
            y += self.splitT[j] @ y
        return y


J = 5
W = LevelWaveletTransformOp(J=J)
#assert np.allclose(W.T_.toarray().T, W.T @ np.eye(2**4 + 1))
plt.imshow(
    np.log10(
        np.abs(W @ np.eye(2**J + 1) - wavelets.WaveletTransformOp(
            J=J, interleaved=True) @ np.eye(2**J + 1))))
plt.colorbar()
plt.show()
