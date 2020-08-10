import numpy as np
import scipy.sparse
from mpi4py import MPI

from mpi_vector import VectorTimeMPI


def as_matrix(operator):
    if hasattr(operator, "A"):
        return operator.A
    cols = operator.shape[1]
    return operator @ np.eye(cols)


class IdentityKronMatMPI:
    def __init__(self, dofs_time, mat_space):
        self.N = dofs_time
        self.M, L = mat_space.shape
        self.mat_space = mat_space
        assert (self.M == L)

    def matvec(self, vec_in, vec_out):
        assert (isinstance(vec_in, VectorTimeMPI))
        assert (self.N == vec_in.N and self.M == vec_in.M)
        assert (vec_in.X_loc.shape == vec_out.X_loc.shape)

        # Apply the local kron.
        vec_out.X_loc = (self.mat_space @ vec_in.X_loc.reshape(
            vec_in.t_end - vec_in.t_begin, vec_in.M).T).T.copy()
        return vec_out


class TridiagKronIdentityMPI:
    def __init__(self, mat_time, dofs_space):
        assert (scipy.sparse.isspmatrix_csr(mat_time))
        self.N, K = mat_time.shape
        self.M = dofs_space
        self.mat_time = mat_time
        assert (self.N == K)
        # Check that it is indeed bandwidthed
        for t, row in enumerate(mat_time):
            assert set(row.indices).issubset(
                set(range(max(0, t - 1), min(t + 2, self.N))))

    def matvec(self, vec_in, vec_out):
        assert (isinstance(vec_in, VectorTimeMPI))
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


class TridiagKronMatMPI:
    def __init__(self, mat_time, mat_space):
        self.mat_time = mat_time
        self.mat_space = mat_space
        N = mat_time.shape[0]
        M = mat_space.shape[0]
        self.I_M = IdentityKronMatMPI(N, mat_space)
        self.T_I = TridiagKronIdentityMPI(mat_time, M)

    def matvec(self, vec_in, vec_out):
        self.I_M.matvec(vec_in, vec_out)
        self.T_I.matvec(vec_out, vec_out)
        return vec_out

    def as_matrix(self):
        return np.kron(as_matrix(self.mat_time), as_matrix(self.mat_space))


if __name__ == "__main__":
    mat = np.array([[3.5, 13., 28.5, 50.,
                     77.5], [-5., -23., -53., -95., -149.],
                    [2.5, 11., 25.5, 46., 72.5]])
    stiff_time = scipy.sparse.spdiags(mat, (1, 0, -1), 5, 5).T.copy().tocsr()
    N = stiff_time.shape[0]
    M = 3
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    print("Hello! I'm rank {} from {}.")

    # Create some global vector on root.
    x_mpi = VectorTimeMPI(comm, N, M)
    x_glob = None
    if rank == 0:
        x_glob = np.random.rand(N * M) * 1.0
        y_glob = np.kron(as_matrix(stiff_time), as_matrix(np.eye(M))) @ x_glob
    x_mpi.scatter(x_glob)

    # Apply the vector using MPI
    T_M = TridiagKronIdentityMPI(stiff_time, M)
    T_M.matvec_inplace(x_mpi)

    # Check that it is corret.
    x_mpi.gather(x_glob)
    if rank == 0:
        assert (np.allclose(y_glob, x_glob))
