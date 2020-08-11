import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg
from scipy import sparse as sp

from mpi_kron import as_matrix


def WaveletTransformMat(J):
    def p(j):
        mat = sp.lil_matrix((2**(j - 1) + 1, 2**j + 1))
        mat[:, 0::2] = sp.eye(2**(j - 1) + 1)
        mat[:, 1::2] = sp.diags([0.5, 0.5], [0, -1],
                                shape=(2**(j - 1) + 1, 2**(j - 1)))
        return mat.T

    def q(j):
        if j == 0: return sp.eye(2)
        mat = sp.lil_matrix((2**(j - 1), 2**j + 1))
        mat[:, 0::2] = sp.diags([-0.5, -0.5], [0, 1],
                                shape=(2**(j - 1), 2**(j - 1) + 1))
        mat[:, 1::2] = sp.eye(2**(j - 1))
        mat[0, 0] = -1
        mat[-1, -1] = -1
        mat *= 2**(j / 2)
        return mat.T

    T = q(0)
    for j in range(1, J + 1):
        T = sp.bmat([[p(j) @ T, q(j)]])

    return T


class WaveletTransformOp(sp.linalg.LinearOperator):
    def __init__(self, J):
        super().__init__(dtype=np.float64, shape=(2**J + 1, 2**J + 1))
        self.J = J
        self.p, self.q = [sp.eye(0)], [sp.eye(2)]
        for j in range(J):
            self.p.append(WaveletTransformOp._p(j + 1))
            self.q.append(WaveletTransformOp._q(j + 1))
        self.pT = [p.T.tocsr() for p in self.p]
        self.qT = [q.T.tocsr() for q in self.q]

    def _p(j):
        mat = sp.lil_matrix((2**(j - 1) + 1, 2**j + 1))
        mat[:, 0::2] = sp.eye(2**(j - 1) + 1)
        mat[:, 1::2] = sp.diags([0.5, 0.5], [0, -1],
                                shape=(2**(j - 1) + 1, 2**(j - 1)))
        return mat.T.tocsr()

    def _q(j):
        if j == 0: return sp.eye(2)
        mat = sp.lil_matrix((2**(j - 1), 2**j + 1))
        mat[:, 0::2] = sp.diags([-0.5, -0.5], [0, 1],
                                shape=(2**(j - 1), 2**(j - 1) + 1))
        mat[:, 1::2] = sp.eye(2**(j - 1))
        mat[0, 0] = -1
        mat[-1, -1] = -1
        mat *= 2**(j / 2)
        return mat.T.tocsr()

    def _matvec(self, x):
        y = x.copy()
        for j in range(1, self.J + 1):
            N_coarse = 2**(j - 1) + 1
            if j == 1:
                N_coarse = 2
            N_fine = 2**j + 1
            y[:N_fine] = self.p[j] @ y[:N_coarse] + self.q[j] @ y[N_coarse:
                                                                  N_fine]
        return y

    def _rmatvec(self, x):
        y = x.copy()
        for j in reversed(range(1, self.J + 1)):
            N_coarse = 2**(j - 1) + 1
            if j == 1:
                N_coarse = 2
            N_fine = 2**j + 1
            y[:N_coarse], y[N_coarse:N_fine] = self.pT[
                j] @ y[:N_fine], self.qT[j] @ y[:N_fine]

        return y


J = 4
WOp = WaveletTransformOp(J)

# Check that wavelettransform is identity for wavelets on the toplevel.
for n, wavelet in enumerate(WOp.qT[J]):
    print(2**J + 1)
    print(n)
    print(as_matrix(wavelet).reshape(-1))
    print(WOp @ as_matrix(wavelet).reshape(-1).T)
