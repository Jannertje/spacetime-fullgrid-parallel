import numpy as np
from mpi_vector import KronVectorMPI
from mpi4py import MPI
import scipy.sparse as sp
import scipy.sparse.linalg
from scipy import sparse as sp

from mpi_kron import CompositeMPI, SparseKronIdentityMPI, as_matrix


def WaveletTransformMat(J):
    """ The matrix transformation from 3-pt wavelet to hat function basis. """
    def p(j):
        mat = sp.dok_matrix((2**(j - 1) + 1, 2**j + 1))
        mat[:, 0::2] = sp.eye(2**(j - 1) + 1)
        mat[:, 1::2] = sp.diags([0.5, 0.5], [0, -1],
                                shape=(2**(j - 1) + 1, 2**(j - 1)))
        return mat.T

    def q(j):
        if j == 0: return sp.eye(2)
        mat = sp.dok_matrix((2**(j - 1), 2**j + 1))
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
    """ A matrix-free transformation from 3-pt wavelet to hat function basis.
    Interleaved means to interleave rows of the p and q matrices.
    """
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
        mat = sp.dok_matrix((2**(j - 1) + 1, 2**j + 1))
        for i in range(mat.shape[0]):
            mat[i, i * 2] = 1
            if i > 0:
                mat[i, i * 2 - 1] = 0.5
            if i + 1 < mat.shape[0]:
                mat[i, i * 2 + 1] = 0.5

        return mat.T.tocsr()

    def _q(j):
        if j == 0: return sp.eye(2)
        mat = sp.dok_matrix((2**(j - 1), 2**j + 1))
        for i in range(mat.shape[0]):
            mat[i, i * 2] = -0.5
            mat[i, i * 2 + 1] = 1
            mat[i, i * 2 + 2] = -0.5
        mat[0, 0] = -1
        mat[-1, -1] = -1

        result = mat.T.tocsr()
        result *= 2**(j / 2)
        return result

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
                y[:N_fine] = self.p[j] @ y[:N_coarse] + self.q[j] @ y[
                    N_coarse:N_fine]
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


class LevelWaveletTransformOp(WaveletTransformOp):
    """ The wavelet transform as composition of J matrices.
    See also eqn (10) from the paper.
    """
    def __init__(self, J):
        super().__init__(J=J, interleaved=True)
        self.split = []
        for j in range(J + 1):
            self.split.append(
                LevelWaveletTransformOp._split(J, j, self.p[j], self.q[j]))

    def _split(J, j, p, q):
        if j == 0: return sp.csr_matrix((2**J + 1, 2**J + 1))

        rows, cols, vals = [], [], []
        S = 2**(J - j)
        p_coo = p.tocoo()
        q_coo = q.tocoo()
        for r, c, v in zip(p_coo.row, p_coo.col, p_coo.data):
            rows.append(S * r)
            cols.append(2 * S * c)
            vals.append(v)
        for r, c, v in zip(q_coo.row, q_coo.col, q_coo.data):
            rows.append(S * r)
            cols.append(S + 2 * S * c)
            vals.append(v)
        for i in range(0, 2**J + 1, S):
            rows.append(i)
            cols.append(i)
            vals.append(-1)

        return sp.csr_matrix((vals, (rows, cols)), shape=(2**J + 1, 2**J + 1))

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
    """ W := W_t kron Id_x. """
    def __init__(self, dofs_distr, J):
        wavelet_transform = LevelWaveletTransformOp(J)
        self.levels = wavelet_transform.levels
        linops = []
        for split in reversed(wavelet_transform.split[1:]):
            linops.append(
                SparseKronIdentityMPI(dofs_distr, split, add_identity=True))
        super().__init__(dofs_distr, linops)


class TransposedWaveletTransformKronIdentityMPI(CompositeMPI):
    """ W.T := W_t.T kron Id_x. """
    def __init__(self, dofs_distr, J):
        wavelet_transform = LevelWaveletTransformOp(J)
        self.levels = wavelet_transform.levels
        linops = []
        for split in wavelet_transform.split[1:]:
            linops.append(
                SparseKronIdentityMPI(dofs_distr,
                                      split.T.tocsr(),
                                      add_identity=True))
        super().__init__(dofs_distr, linops)


if __name__ == "__main__":
    """ A quick test. """
    J_time = 9
    M = 50
    N = 513
    start_time = MPI.Wtime()
    W = WaveletTransformKronIdentityMPI(J_time, M)
    WT = TransposedWaveletTransformKronIdentityMPI(J_time, M)
    vec = KronVectorMPI(MPI.COMM_WORLD, N, M)
    vec.X_loc = np.random.rand(*vec.X_loc.shape)
    if MPI.COMM_WORLD.rank == 0:
        print('MPI Tasks:', MPI.COMM_WORLD.Get_size())

    MPI.COMM_WORLD.Barrier()
    start_time = MPI.Wtime()
    for _ in range(10):
        W @ vec
    if MPI.COMM_WORLD.rank == 0:
        print('W: ', MPI.Wtime() - start_time)

    MPI.COMM_WORLD.Barrier()
    start_time = MPI.Wtime()
    for _ in range(10):
        WT @ vec
    if MPI.COMM_WORLD.rank == 0:
        print('WT:', MPI.Wtime() - start_time)
