import numpy as np
import traceback
import scipy.sparse

from mpi4py import MPI
from mpi_vector import KronVectorMPI


def as_matrix(operator):
    cols = operator.shape[1]
    return operator @ np.eye(cols)


class LinearOperatorMPI:
    def __init__(self, dofs_distr):
        self.dofs_distr = dofs_distr
        self.N = dofs_distr.N
        self.M = dofs_distr.M
        self.num_applies = 0
        self.time_applies = 0
        self.time_communication = 0

    def __matmul__(self, x):
        assert isinstance(x, KronVectorMPI)
        start_time = MPI.Wtime()

        y = self._matvec(x, KronVectorMPI(self.dofs_distr))

        self.num_applies += 1
        self.time_applies += MPI.Wtime() - start_time
        return y

    def time_per_apply(self):
        assert (self.time_applies)
        return self.time_applies / self.num_applies, self.time_communication / self.num_applies

    def as_global_matrix(self):
        print('Expensive computation!')
        I = np.eye(self.N * self.M)
        comm = MPI.COMM_WORLD
        x_mpi = KronVectorMPI(self.dofs_distr)
        result = None
        x_glob = None
        if x_mpi.rank == 0:
            x_glob = np.empty(self.N * self.M)
            result = np.zeros((self.N * self.M, self.N * self.M))

        for k in range(self.N * self.M):
            x_mpi.scatter(I[k, :])

            # Apply the operator using MPI.
            x_mpi = self @ x_mpi

            # Store the results.
            x_mpi.gather(x_glob)
            if x_mpi.rank == 0:
                result[:, k] = x_glob
        return result


class IdentityMPI(LinearOperatorMPI):
    def __init__(self, dofs_distr):
        super().__init__(dofs_distr)

    def _matvec(self, vec_in, vec_out):
        vec_out.X_loc[:] = vec_in.X_loc
        return vec_out


class SumMPI(LinearOperatorMPI):
    def __init__(self, dofs_distr, linops):
        assert all(isinstance(linop, LinearOperatorMPI) for linop in linops)
        N, M = linops[0].N, linops[0].M
        self.linops = linops
        super().__init__(dofs_distr)

    def _matvec(self, vec_in, vec_out):
        assert (vec_in is not vec_out)
        self.time_communication = 0

        vec_in_cpy = vec_in.X_loc.copy()

        vec_out.reset()
        vec_tmp = KronVectorMPI(self.dofs_distr)

        for linop in self.linops:
            linop._matvec(vec_in, vec_tmp)
            vec_out += vec_tmp
            self.time_communication += linop.time_communication

        assert np.all(vec_in_cpy == vec_in.X_loc)

        return vec_out


class CompositeMPI(LinearOperatorMPI):
    def __init__(self, dofs_distr, linops):
        assert all(isinstance(linop, LinearOperatorMPI) for linop in linops)
        N, M = linops[0].N, linops[0].M
        assert all(linop.N == N and linop.M == M for linop in linops)
        self.linops = linops
        super().__init__(dofs_distr)

    def _matvec(self, vec_in, vec_out):
        assert (vec_in is not vec_out)
        self.time_communication = 0
        vec_out.X_loc = None
        Y = vec_in
        for linop in reversed(self.linops):
            Y = linop @ Y
            self.time_communication += linop.time_communication
        vec_out.X_loc = Y.X_loc
        return vec_out


class BlockDiagMPI(LinearOperatorMPI):
    def __init__(self, dofs_distr, matrices_space):
        N = len(matrices_space)
        M = matrices_space[0].shape[0]
        for mat in matrices_space:
            assert mat.shape == (M, M)
        self.matrices_space = matrices_space
        super().__init__(dofs_distr)

    def _matvec(self, vec_in, vec_out):
        assert (isinstance(vec_in, KronVectorMPI))
        assert (self.N == vec_in.N and self.M == vec_in.M)
        assert (vec_in.X_loc.shape == vec_out.X_loc.shape)
        assert (vec_out is not vec_in)

        # Apply the operators on the diagonal.
        for t_loc, linop in enumerate(
                self.matrices_space[vec_in.t_begin:vec_in.t_end]):
            vec_out.X_loc[t_loc] = (linop @ vec_in.X_loc[t_loc].T).T
        return vec_out


class IdentityKronMatMPI(LinearOperatorMPI):
    def __init__(self, dofs_distr, mat_space):
        M, L = mat_space.shape
        assert (M == L)
        self.mat_space = mat_space
        super().__init__(dofs_distr)

    def _matvec(self, vec_in, vec_out):
        assert (isinstance(vec_in, KronVectorMPI))
        assert (self.N == vec_in.N and self.M == vec_in.M)
        assert (vec_in.X_loc.shape == vec_out.X_loc.shape)

        # Apply the local kron.
        vec_out.X_loc[:] = (self.mat_space @ vec_in.X_loc.T).T
        return vec_out


class TridiagKronIdentityMPI(LinearOperatorMPI):
    def __init__(self, dofs_distr, mat_time):
        assert (scipy.sparse.isspmatrix_csr(mat_time))
        N, K = mat_time.shape
        M = dofs_distr.M
        assert (N == K)
        # Check that it is indeed bandwidthed
        for t, row in enumerate(mat_time):
            assert set(row.indices).issubset(
                set(range(max(0, t - 1), min(t + 2, N))))

        coo_mat = mat_time.tocoo()

        # Create a sliced up matrix.
        row, col, val = [], [], []
        for r, c, v in zip(coo_mat.row, coo_mat.col, coo_mat.data):
            if dofs_distr.t_begin <= r < dofs_distr.t_end:
                row.append(r - dofs_distr.t_begin)
                col.append(c - dofs_distr.t_begin + 1)
                val.append(v)

        sliced_mat = scipy.sparse.csr_matrix(
            (val, (row, col)),
            shape=(dofs_distr.t_end - dofs_distr.t_begin,
                   dofs_distr.t_end - dofs_distr.t_begin + 2))

        # Now turn this matrix apart in 3 parts.
        self.inner_mat = sliced_mat[1:-1, :]
        self.top_row = sliced_mat[0, :]
        self.bottom_row = sliced_mat[-1, :]
        super().__init__(dofs_distr)

    def _matvec(self, vec_in, vec_out):
        assert (isinstance(vec_in, KronVectorMPI))
        assert (self.N == vec_in.N and self.M == vec_in.M)
        assert (vec_in.X_loc.shape == vec_out.X_loc.shape)
        assert (vec_in is not vec_out)

        # Communicate.
        def callback():
            vec_out.X_loc[1:-1] = self.inner_mat @ vec_in.X_loc_bdr

        self.time_communication += vec_in.communicate_bdr(callback)

        # Apply top/bottom rows
        vec_out.X_loc[0, :] = self.top_row @ vec_in.X_loc_bdr
        vec_out.X_loc[-1, :] = self.bottom_row @ vec_in.X_loc_bdr
        return vec_out


class TridiagKronMatMPI(LinearOperatorMPI):
    def __init__(self, dofs_distr, mat_time, mat_space):
        N = mat_time.shape[0]
        M = mat_space.shape[0]
        super().__init__(dofs_distr)
        self.mat_time = mat_time
        self.mat_space = mat_space
        self.I_M = IdentityKronMatMPI(dofs_distr, mat_space)
        self.T_I = TridiagKronIdentityMPI(dofs_distr, mat_time)

    def _matvec(self, vec_in, vec_out):
        vec_tmp = KronVectorMPI(self.dofs_distr)

        self.T_I._matvec(vec_in, vec_tmp)
        self.I_M._matvec(vec_tmp, vec_out)
        self.time_communication = self.I_M.time_communication + self.T_I.time_communication
        return vec_out

    def as_matrix(self):
        return np.kron(as_matrix(self.mat_time), as_matrix(self.mat_space))


class MatKronIdentityMPI(LinearOperatorMPI):
    """
    This applies a M kron I by swapping the rearanging the input vector.
    NOTE: Expensive in communication!.
    """
    def __init__(self, dofs_distr, mat_time):
        N, K = mat_time.shape
        M = dofs_distr.N
        assert (N == K)
        self.mat_time = mat_time
        if hasattr(mat_time, 'levels'):
            self.levels = mat_time.levels
        super().__init__(dofs_distr)

    def _matvec(self, vec_in, vec_out):
        assert (isinstance(vec_in, KronVectorMPI))
        assert (self.N == vec_in.N and self.M == vec_in.M)
        assert (vec_in.X_loc.shape == vec_out.X_loc.shape)

        # First permute the vector.
        vec_in_permuted, comm_time = vec_in.permute()
        self.time_communication += comm_time

        # Apply the local kron.
        vec_in_permuted.X_loc[:] = (self.mat_time @ vec_in_permuted.X_loc.T).T

        # Permute the vector back.
        _, comm_time = vec_in_permuted.permute(vec_out)
        self.time_communication += comm_time

        return vec_out


class SparseKronIdentityMPI(LinearOperatorMPI):
    """ Requires a square matrix in CSR with symmetric sparsity pattern. """
    def __init__(self, dofs_distr, mat_time, add_identity=False):
        super().__init__(dofs_distr)
        assert scipy.sparse.isspmatrix_csr(mat_time)
        N, K = mat_time.shape
        M = dofs_distr.N
        assert (N == K)
        assert (mat_time.nnz)
        self.add_identity = add_identity
        coo_mat = mat_time.tocoo()
        sliced_mat = list(
            filter(lambda rdc: dofs_distr.t_begin <= rdc[0] < dofs_distr.t_end,
                   zip(coo_mat.row, coo_mat.col, coo_mat.data)))
        if sliced_mat:
            self.row, self.col, self.data = zip(*sliced_mat)
        else:
            self.row = self.col = self.data = []

        self.comm_dofs = sorted(
            set((row, col) for row, col in zip(self.row, self.col)
                if col < dofs_distr.t_begin or dofs_distr.t_end <= col))

    def _matvec(self, vec_in, vec_out):
        assert (isinstance(vec_in, KronVectorMPI))
        assert (self.N == vec_in.N and self.M == vec_in.M)
        assert (vec_in.X_loc.shape == vec_out.X_loc.shape)

        # Start up communication.
        if len(self.comm_dofs):
            X_recv, reqs = vec_in.communicate_dofs(self.comm_dofs)

        if self.add_identity:
            assert vec_out is not vec_in
            vec_out.X_loc[:] = vec_in.X_loc
        else:
            vec_out.reset()

        # Wait for communication to complete.
        if len(self.comm_dofs):
            start_time = MPI.Wtime()
            MPI.Request.Waitall(reqs)
            self.time_communication += MPI.Wtime() - start_time

        # for every processor, loop over its own timesteps
        for t, idx, coeff in zip(self.row, self.col, self.data):
            t_ = t - vec_in.t_begin
            idx_ = idx - vec_in.t_begin
            if vec_in.t_begin <= idx < vec_in.t_end:
                # The data is in X_loc
                vec_out.X_loc[t_, :] += coeff * vec_in.X_loc[idx_]
            else:

                vec_out.X_loc[t_, :] += coeff * X_recv[idx]

        return vec_out
