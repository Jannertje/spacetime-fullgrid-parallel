import numpy as np

from mpi_vector import VectorTimeMPI


class IdentityKronMatMPI:
    def __init__(self, size_time, mat_space):
        self.N = size_time
        self.M, L = mat_space.shape
        self.mat_space = mat_space
        assert (self.M == L)

    def matvec_inplace(self, vec):
        assert (isinstance(vec, VectorTimeMPI))
        assert (self.N == vec.N and self.M == vec.M)

        # Apply the local kron.
        vec.X_loc = (self.mat_space @ vec.X_loc.reshape(
            vec.t_end - vec.t_begin, vec.M).T).T.copy()


class TridiagKronIdentityMPI:
    def __init__(self, mat_time, size_space):
        self.N, K = mat_time.shape
        self.M = size_space
        self.mat_time = mat_time
        assert (self.N == K)
        # Check that it is indeed bandwidthed
        for t, row in enumerate(mat_time):
            assert np.all(row.indices == np.arange(max(0, t -
                                                       1), min(t + 2, self.N)))

    def matvec_inplace(self, vec):
        assert (isinstance(vec, VectorTimeMPI))
        assert (self.N == vec.N and self.M == vec.M)

        bdr = vec.communicate_bdr()

        X_loc_bdr = np.vstack([bdr[0], vec.X_loc, bdr[-1]])
        Y_loc = np.empty(vec.X_loc.shape, dtype=np.float64)

        # for every processor, loop over its own timesteps
        for t in range(vec.t_begin, vec.t_end):
            row = self.mat_time[t, :].data
            begin_slice = t - vec.t_begin if t > 0 else 1
            Y_loc[t - vec.t_begin, :] = row @ X_loc_bdr[begin_slice:(
                begin_slice + row.shape[0]), :]

        vec.X_loc = Y_loc


class TridiagKronMatMPI:
    def __init__(self, mat_time, mat_space):
        N = mat_time.shape[0]
        M = mat_space.shape[0]
        self.I_M = IdentityKronMatMPI(N, mat_space)
        self.T_I = TridiagKronIdentityMPI(mat_time, M)

    def matvec_inplace(self, vec):
        self.I_M.matvec_inplace(vec)
        self.T_I.matvec_inplace(vec)
