import numpy as np
import scipy.sparse
from mpi4py import MPI

from mpi_vector import KronVectorMPI


def as_matrix(operator):
    if hasattr(operator, "A"):
        return operator.A
    cols = operator.shape[1]
    return operator @ np.eye(cols)


class LinearOperatorMPI:
    def __init__(self, dofs_time, dofs_space):
        self.N = dofs_time
        self.M = dofs_space

    def __matmul__(self, x):
        assert isinstance(x, KronVectorMPI)
        y = KronVectorMPI(x.comm, x.N, x.M)
        return self.matvec(x, y)


class IdentityMPI(LinearOperatorMPI):
    def __init__(self, dofs_time, dofs_space):
        super().__init__(dofs_time, dofs_space)

    def matvec(self, vec_in, vec_out):
        vec_out.X_loc[:] = vec_in.X_loc
        return vec_out


class IdentityKronMatMPI(LinearOperatorMPI):
    def __init__(self, dofs_time, mat_space):
        M, L = mat_space.shape
        assert (M == L)
        self.mat_space = mat_space
        super().__init__(dofs_time, M)

    def matvec(self, vec_in, vec_out):
        assert (isinstance(vec_in, KronVectorMPI))
        assert (self.N == vec_in.N and self.M == vec_in.M)
        assert (vec_in.X_loc.shape == vec_out.X_loc.shape)

        # Apply the local kron.
        vec_out.X_loc[:] = (self.mat_space @ vec_in.X_loc.T).T
        return vec_out


class MatKronIdentityMPI(LinearOperatorMPI):
    def __init__(self, mat_time, dofs_space):
        N, K = mat_time.shape
        M = dofs_space
        assert (N == K)
        self.mat_time = mat_time
        super().__init__(N, M)

    def matvec(self, vec_in, vec_out):
        assert (isinstance(vec_in, KronVectorMPI))
        assert (self.N == vec_in.N and self.M == vec_in.M)
        assert (vec_in.X_loc.shape == vec_out.X_loc.shape)

        # First permute the vector.
        vec_in_permuted = vec_in.permute()

        # Apply the local kron.
        vec_in_permuted.X_loc[:] = (self.mat_time @ vec_in_permuted.X_loc.T).T

        # Permute the vector back.
        vec_in_permuted.permute(vec_out)

        return vec_out


class TridiagKronIdentityMPI(LinearOperatorMPI):
    def __init__(self, mat_time, dofs_space):
        assert (scipy.sparse.isspmatrix_csr(mat_time))
        N, K = mat_time.shape
        M = dofs_space
        assert (N == K)
        self.mat_time = mat_time
        # Check that it is indeed bandwidthed
        for t, row in enumerate(mat_time):
            assert set(row.indices).issubset(
                set(range(max(0, t - 1), min(t + 2, N))))

        super().__init__(N, M)

    def matvec(self, vec_in, vec_out):
        assert (isinstance(vec_in, KronVectorMPI))
        assert (self.N == vec_in.N and self.M == vec_in.M)
        assert (vec_in.X_loc.shape == vec_out.X_loc.shape)

        bdr = vec_in.communicate_bdr()

        X_loc_bdr = np.vstack([bdr[0], vec_in.X_loc, bdr[-1]])
        Y_loc = np.zeros(vec_out.X_loc.shape, dtype=np.float64)

        # for every processor, loop over its own timesteps
        for t in range(vec_in.t_begin, vec_in.t_end):
            row = self.mat_time[t, :]
            for idx, coeff in zip(row.indices, row.data):
                Y_loc[t - vec_out.t_begin, :] += coeff * X_loc_bdr[
                    idx - vec_in.t_begin + 1]

        vec_out.X_loc = Y_loc
        return vec_out


class TridiagKronMatMPI(LinearOperatorMPI):
    def __init__(self, mat_time, mat_space):
        N = mat_time.shape[0]
        M = mat_space.shape[0]
        super().__init__(N, M)
        self.mat_time = mat_time
        self.mat_space = mat_space
        self.I_M = IdentityKronMatMPI(N, mat_space)
        self.T_I = TridiagKronIdentityMPI(mat_time, M)

    def matvec(self, vec_in, vec_out):
        self.I_M.matvec(vec_in, vec_out)
        self.T_I.matvec(vec_out, vec_out)
        return vec_out

    def as_matrix(self):
        return np.kron(as_matrix(self.mat_time), as_matrix(self.mat_space))
