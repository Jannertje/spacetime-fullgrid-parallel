import numpy as np
from mpi4py import MPI


class KronVectorMPI:
    """ This class represents a vector that is parallized in the first
    component (e.g., time).
    """
    def __init__(self, comm, N, M):
        self.N = N
        self.M = M
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        assert (self.N >= self.size)

        # Divide the dofs among the proessors.
        block_size = self.N // self.size
        rest_size = self.N % self.size
        self.dof_distribution = []
        counter = 0
        self.displs = np.empty(self.size)
        self.counts = np.empty(self.size)
        for p in range(self.size):
            self.displs[p] = counter * self.M
            self.dof_distribution.append([counter, counter + block_size])
            if self.size - p - 1 < rest_size:
                self.dof_distribution[-1][1] += 1
            counter = self.dof_distribution[-1][1]
            self.counts[p] = (self.dof_distribution[-1][1] -
                              self.dof_distribution[-1][0]) * self.M

        assert (counter == self.N)
        self.t_begin, self.t_end = self.dof_distribution[self.rank]

        # Create a local vector containing the dofs.
        self.X_loc = np.zeros((self.t_end - self.t_begin, self.M),
                              dtype=np.float64)

    def __iadd__(self, other):
        self.X_loc += other.X_loc
        return self

    def __add__(self, other):
        vec_out = KronVectorMPI(self.comm, self.N, self.M)
        vec_out.X_loc = self.X_loc + other.X_loc
        return vec_out

    def __isub__(self, other):
        self.X_loc -= other.X_loc
        return self

    def __sub__(self, other):
        vec_out = KronVectorMPI(self.comm, self.N, self.M)
        vec_out.X_loc = self.X_loc - other.X_loc
        return vec_out

    def __rmul__(self, other):
        vec_out = KronVectorMPI(self.comm, self.N, self.M)
        vec_out.X_loc = other * self.X_loc
        return vec_out

    def dof2proc(self):
        result = np.zeros(self.N)
        for p, (t_begin, t_end) in enumerate(self.dof_distribution):
            for dof in range(t_begin, t_end):
                result[dof] = p
        return result

    def scatter(self, X_glob):
        data = [None, None, None, MPI.DOUBLE]
        if self.rank == 0:
            data = [X_glob, self.counts, self.displs, MPI.DOUBLE]

        self.comm.Scatterv(data, self.X_loc)

    def gather(self, X_glob):
        self.comm.Gatherv(self.X_loc,
                          [X_glob, self.counts, self.displs, MPI.DOUBLE])

    def communicate_bdr(self):
        if self.rank > 0:
            self.comm.Isend(self.X_loc[0, :], self.rank - 1)
        if self.rank + 1 < self.size:
            self.comm.Isend(self.X_loc[-1, :], self.rank + 1)

        bdr = [np.zeros(self.M), np.zeros(self.M)]
        if self.rank > 0:
            self.comm.Recv(bdr[0], source=self.rank - 1)
        if self.rank + 1 < self.size:
            self.comm.Recv(bdr[1], source=self.rank + 1)

        return bdr

    def dot(self, vec_other):
        assert (isinstance(vec_other, KronVectorMPI))
        assert (vec_other.X_loc.shape == self.X_loc.shape)
        dot_loc = np.dot(self.X_loc.reshape(-1), vec_other.X_loc.reshape(-1))
        dot_glob = self.comm.allreduce(dot_loc)
        return dot_glob

    def permute(self, vec_perm=None):
        """ Permutes the order of the kronecker product. """
        if vec_perm is None:
            vec_perm = KronVectorMPI(self.comm, self.M, self.N)
        else:
            assert (vec_perm.N == self.M and vec_perm.M == self.N)

        # Now lets send all the dofs.
        for p in range(self.size):
            x_begin, x_end = vec_perm.dof_distribution[p]
            self.comm.Isend(self.X_loc[:, x_begin:x_end].copy(), dest=p)

        # Receive all the dofs.
        for p in range(self.size):
            t_begin, t_end = self.dof_distribution[p]
            x_begin, x_end = vec_perm.t_begin, vec_perm.t_end
            buf = np.empty((t_end - t_begin, x_end - x_begin))
            self.comm.Recv(buf, source=p)
            vec_perm.X_loc[:, t_begin:t_end] = buf.T
        return vec_perm
