import numpy as np
import scipy.sparse

from mpi4py import MPI
from mpi_vector import KronVectorMPI


def as_matrix(operator):
    cols = operator.shape[1]
    return operator @ np.eye(cols)


class LinearOperatorMPI:
    def __init__(self, dofs_time, dofs_space):
        self.N = dofs_time
        self.M = dofs_space
        self.num_applies = 0
        self.time_applies = 0

    def __matmul__(self, x):
        assert isinstance(x, KronVectorMPI)
        start_time = MPI.Wtime()

        y = self._matvec(x, KronVectorMPI(x.comm, x.N, x.M))

        self.num_applies += 1
        self.time_applies += MPI.Wtime() - start_time
        return y

    def time_per_apply(self):
        assert (self.time_applies)
        return self.time_applies / self.num_applies

    def as_global_matrix(self):
        print('Expensive computation!')
        I = np.eye(self.N * self.M)
        comm = MPI.COMM_WORLD
        x_mpi = KronVectorMPI(comm, self.N, self.M)
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

    def as_global_transposed_matrix(self):
        print('Expensive computation!')
        I = np.eye(self.N * self.M)
        comm = MPI.COMM_WORLD
        x_mpi = KronVectorMPI(comm, self.N, self.M)
        result = None
        x_glob = None
        if x_mpi.rank == 0:
            x_glob = np.empty(self.N * self.M)
            result = np.zeros((self.N * self.M, self.N * self.M))

        for k in range(self.N * self.M):
            x_mpi.scatter(I[k, :])

            # Apply the operator using MPI.
            x_mpi = self.rmatvec(x_mpi)

            # Store the results.
            x_mpi.gather(x_glob)
            if x_mpi.rank == 0:
                result[:, k] = x_glob
        return result


class IdentityMPI(LinearOperatorMPI):
    def __init__(self, dofs_time, dofs_space):
        super().__init__(dofs_time, dofs_space)

    def _matvec(self, vec_in, vec_out):
        vec_out.X_loc[:] = vec_in.X_loc
        return vec_out


class SumMPI(LinearOperatorMPI):
    def __init__(self, linops):
        assert all(isinstance(linop, LinearOperatorMPI) for linop in linops)
        N, M = linops[0].N, linops[0].M
        self.linops = linops
        super().__init__(N, M)

    def _matvec(self, vec_in, vec_out):
        assert (vec_in is not vec_out)
        Y_loc = np.zeros(vec_out.X_loc.shape)

        for linop in self.linops:
            linop._matvec(vec_in, vec_out)
            Y_loc += vec_out.X_loc

        vec_out.X_loc = Y_loc
        return vec_out


class CompositeMPI(LinearOperatorMPI):
    def __init__(self, linops):
        assert all(isinstance(linop, LinearOperatorMPI) for linop in linops)
        N, M = linops[0].N, linops[0].M
        assert all(linop.N == N and linop.M == M for linop in linops)
        self.linops = linops
        super().__init__(N, M)

    def _matvec(self, vec_in, vec_out):
        assert (vec_in is not vec_out)
        Y = vec_in
        for linop in reversed(self.linops):
            Y = linop @ Y
        vec_out.X_loc = Y.X_loc
        return vec_out

    def _rmatvec(self, vec_in, vec_out):
        assert (vec_in is not vec_out)
        Y = vec_in
        for linop in self.linops:
            Y = linop.rmatvec(Y)
        vec_out.X_loc = Y.X_loc
        return vec_out


class BlockDiagMPI(LinearOperatorMPI):
    def __init__(self, matrices_space):
        N = len(matrices_space)
        M = matrices_space[0].shape[0]
        for mat in matrices_space:
            assert mat.shape == (M, M)
        self.matrices_space = matrices_space
        super().__init__(N, M)

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
    def __init__(self, dofs_time, mat_space):
        M, L = mat_space.shape
        assert (M == L)
        self.mat_space = mat_space
        super().__init__(dofs_time, M)

    def _matvec(self, vec_in, vec_out):
        assert (isinstance(vec_in, KronVectorMPI))
        assert (self.N == vec_in.N and self.M == vec_in.M)
        assert (vec_in.X_loc.shape == vec_out.X_loc.shape)

        # Apply the local kron.
        vec_out.X_loc[:] = (self.mat_space @ vec_in.X_loc.T).T
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

    def _matvec(self, vec_in, vec_out):
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

    def _matvec(self, vec_in, vec_out):
        self.I_M._matvec(vec_in, vec_out)
        self.T_I._matvec(vec_out, vec_out)
        return vec_out

    def as_matrix(self):
        return np.kron(as_matrix(self.mat_time), as_matrix(self.mat_space))


class MatKronIdentityMPI(LinearOperatorMPI):
    """
    This applies a M kron I by swapping the rearanging the input vector.
    NOTE: Expensive in communication!.
    """
    def __init__(self, mat_time, dofs_space):
        N, K = mat_time.shape
        M = dofs_space
        assert (N == K)
        self.mat_time = mat_time
        super().__init__(N, M)

    def _matvec(self, vec_in, vec_out):
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


class SparseKronIdentityMPI(LinearOperatorMPI):
    """ Requires a square matrix in CSR with symmetric sparsity pattern. """
    def __init__(self, mat_time, dofs_space, add_identity=False):
        assert (scipy.sparse.isspmatrix_csr(mat_time))
        N, K = mat_time.shape
        M = dofs_space
        assert (N == K)
        self.mat_time = mat_time
        self.comm_dofs = None
        self.add_identity = add_identity
        super().__init__(N, M)

    def _matvec(self, vec_in, vec_out):
        assert (isinstance(vec_in, KronVectorMPI))
        assert (self.N == vec_in.N and self.M == vec_in.M)
        assert (vec_in.X_loc.shape == vec_out.X_loc.shape)

        if not self.comm_dofs:
            comm_dofs = set()
            for row in range(vec_in.t_begin, vec_in.t_end):
                for col in self.mat_time[row, :].indices:
                    if col < vec_in.t_begin or vec_in.t_end <= col:
                        comm_dofs.add((row, col))
            self.comm_dofs = sorted(comm_dofs)
        recv_dofs = [dof for _, dof in self.comm_dofs]

        X = scipy.sparse.lil_matrix((vec_in.N, vec_in.M))
        X[vec_in.t_begin:vec_in.t_end] = vec_in.X_loc
        if len(self.comm_dofs):
            X[recv_dofs] = vec_in.communicate_dofs(self.comm_dofs)

        # TODO: replace dense matmat
        Y = (self.mat_time[vec_in.t_begin:vec_in.t_end] @ X.tocsc()).toarray()
        if self.add_identity:
            vec_out.X_loc = vec_in.X_loc + Y
        else:
            vec_out.X_loc = Y

        return vec_out
