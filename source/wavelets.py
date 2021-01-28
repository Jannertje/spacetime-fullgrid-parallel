import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg
from mpi4py import MPI
from scipy import sparse as sp

from .mpi_kron import CompositeMPI, SparseKronIdentityMPI, as_matrix
from .mpi_vector import KronVectorMPI


def WaveletTransformMat(J):
    """ The matrix transformation from 3-pt wavelet to hat function basis.

    The wavelet indices are ordered level-by-level, i.e., for d_k the wavelet
    coordinates on level k, the input vector is ordered as[d_0, d_1, ..., d_J].

    NOTE: This is a debug function; the resulting matrix is not uniformly 
          sparse, and therefore cannot be applied in linear complexity.
    """
    def p(j):
        """ Prolongate hat function of level j-1 to level j. """
        mat = sp.dok_matrix((2**(j - 1) + 1, 2**j + 1))
        mat[:, 0::2] = sp.eye(2**(j - 1) + 1)
        mat[:, 1::2] = sp.diags([0.5, 0.5], [0, -1],
                                shape=(2**(j - 1) + 1, 2**(j - 1)))
        return mat.T

    def q(j):
        """ Embed 3-pt wavelets on level j into hat functions on level j. """
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

    By iterating over the levels, this applies the `splitting' 
    operation [p(j), q(j)] in-place in linear complexity.  In view 
    of parallelism, this implements the Wavelet transform using two
    different indexing strategies. 

    For interleaved=false, the `canonical' level-by-level indexing is used,
    i.e., the wavelet vector is ordered as[d_0, d_1, ..., d_J].
    
    For interleaved=true, the 3-pt wavelets are numbered according
    to the associated nodes on the mesh of level J. For example, if J=2,
    then the wavelets are ordered as [lvl_0, lvl_2, lvl_1, lvl_2, lvl_0].
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

    def _matmat(self, X):
        Y = X.copy()
        if self.interleaved:
            for j in range(1, self.J + 1):
                S = 2**(self.J - j)
                Y[::S] = self.p[j] @ Y[::S][::2] + self.q[j] @ Y[::S][1::2]
        else:
            for j in range(1, self.J + 1):
                N_coarse = 2**(j - 1) + 1
                N_fine = 2**j + 1
                Y[:N_fine] = self.p[j] @ Y[:N_coarse] + self.q[j] @ Y[
                    N_coarse:N_fine]
        return Y

    def _rmatmat(self, X):
        Y = X.copy()
        if self.interleaved:
            for j in reversed(range(1, self.J + 1)):
                S = 2**(self.J - j)
                Y[::S][::2], Y[::S][
                    1::2] = self.pT[j] @ Y[::S], self.qT[j] @ Y[::S]
        else:
            for j in reversed(range(1, self.J + 1)):
                N_coarse = 2**(j - 1) + 1
                N_fine = 2**j + 1
                Y[:N_coarse], Y[N_coarse:N_fine] = self.pT[
                    j] @ Y[:N_fine], self.qT[j] @ Y[:N_fine]

        return Y

    def split(self, j):
        """ Helper function that represents operation on level j as a matrix.

        This helper function is used to return the matrix [p(j) q(j)]
        when interleaved=true, as this corresponds to a matrix with rows/columns
        of p(j) and q(j) interleaved. 

        NOTE: In order to achieve a linear complexity algorithm, this matrix
              should be applied in-place (or combined with an additional 
              identity matrix to copy the input vector).
        """
        J = self.J
        p = self.p[j]
        q = self.q[j]
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


class WaveletTransformKronIdentityMPI(CompositeMPI):
    """ W := W_t kron Id_x. """
    def __init__(self, dofs_distr, J):
        wavelet_transform = WaveletTransformOp(J, interleaved=True)
        self.levels = wavelet_transform.levels
        linops = []
        for j in reversed(range(1, J + 1)):
            split_mat = wavelet_transform.split(j)
            linops.append(
                SparseKronIdentityMPI(dofs_distr, split_mat,
                                      add_identity=True))
        super().__init__(dofs_distr, linops)


class TransposedWaveletTransformKronIdentityMPI(CompositeMPI):
    """ W.T := W_t.T kron Id_x. """
    def __init__(self, dofs_distr, J):
        wavelet_transform = WaveletTransformOp(J, interleaved=True)
        self.levels = wavelet_transform.levels
        linops = []
        for j in range(1, J + 1):
            split_mat = wavelet_transform.split(j)
            linops.append(
                SparseKronIdentityMPI(dofs_distr,
                                      split_mat.T.tocsr(),
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
