import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg
from scipy import sparse as sp

from mpi_kron import CompositeMPI, SparseKronIdentityMPI, as_matrix


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


class TransposeLinearOp(sp.linalg.LinearOperator):
    def __init__(self, linop):
        super().__init__(dtype=np.float64,
                         shape=(linop.shape[1], linop.shape[0]))
        self.linop = linop

    def _matmat(self, x):
        return self.linop._rmatmat(x)


class WaveletTransformOp(sp.linalg.LinearOperator):
    def __init__(self, J, interleaved=False):
        super().__init__(dtype=np.float64, shape=(2**J + 1, 2**J + 1))
        self.J = J
        self.p, self.q = [sp.eye(1)], [sp.eye(2)]
        for j in range(J):
            self.p.append(WaveletTransformOp._p(j + 1))
            self.q.append(WaveletTransformOp._q(j + 1))
        self.pT = [p.T.tocsr() for p in self.p]
        self.qT = [q.T.tocsr() for q in self.q]
        self.interleaved = interleaved
        if interleaved:
            self.levels = np.zeros(2**J + 1, dtype=int)
            for j in reversed(range(0, J + 1)):
                self.levels[::2**(J - j)] = j
        else:
            N_wavelets_lvl = [2] + [2**(j - 1) for j in range(1, self.J + 1)]
            self.levels = [
                j for j, N_wavelets in enumerate(N_wavelets_lvl)
                for _ in range(N_wavelets)
            ]

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

    def _matmat(self, x):
        y = x.copy()
        if self.interleaved:
            for j in range(1, self.J + 1):
                S = 2**(self.J - j)
                y[::S] = self.p[j] @ y[::S][::2] + self.q[j] @ y[::S][1::2]
        else:
            for j in range(1, self.J + 1):
                N_coarse = 2**(j - 1) + 1
                N_fine = 2**j + 1
                y[:N_fine] = self.p[j] @ y[:N_coarse] + self.q[j] @ y[N_coarse:
                                                                      N_fine]
        return y

    def _rmatmat(self, x):
        y = x.copy()
        if self.interleaved:
            for j in reversed(range(1, self.J + 1)):
                S = 2**(self.J - j)
                y[::S][::2], y[::S][
                    1::2] = self.pT[j] @ y[::S], self.qT[j] @ y[::S]
        else:
            for j in reversed(range(1, self.J + 1)):
                N_coarse = 2**(j - 1) + 1
                N_fine = 2**j + 1
                y[:N_coarse], y[N_coarse:N_fine] = self.pT[
                    j] @ y[:N_fine], self.qT[j] @ y[:N_fine]

        return y

    def _adjoint(self):
        return TransposeLinearOp(self)


class LevelWaveletTransformOp(WaveletTransformOp):
    def __init__(self, J):
        super().__init__(J=J, interleaved=True)
        self.split = []
        for j in range(J + 1):
            self.split.append(
                LevelWaveletTransformOp._split(J, j, self.p[j], self.q[j]))

    def _split(J, j, p, q):
        if j == 0: return sp.csr_matrix((2**J + 1, 2**J + 1))
        I = sp.eye(2**J + 1, format='csr')
        mat = sp.eye(2**J + 1, format='csr')
        S = 2**(J - j)
        mat[::S] = p @ I[::S][::2] + q @ I[::S][1::2]
        mat -= sp.eye(2**J + 1, format='csr')
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


class WaveletTransformKronIdentityMPI(CompositeMPI):
    def __init__(self, J, M):
        wavelet_transform = LevelWaveletTransformOp(J)
        linops = []
        for split in reversed(wavelet_transform.split):
            linops.append(SparseKronIdentityMPI(split, M, add_identity=True))
        super().__init__(linops)
