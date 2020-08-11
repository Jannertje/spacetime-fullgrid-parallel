import numpy as np
from mpi4py import MPI


class VectorTimeMPI:
    def __init__(self, comm, dofs_time, dofs_space):
        self.N = dofs_time
        self.M = dofs_space
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        assert (self.N >= self.size)

        # Divide the dofs among the proessors.
        block_size = self.N // self.size
        rest_size = self.N % self.size
        dof_distribution = []
        counter = 0
        self.displs = np.empty(self.size)
        self.counts = np.empty(self.size)
        for p in range(self.size):
            self.displs[p] = counter * self.M
            dof_distribution.append([counter, counter + block_size])
            if self.size - p - 1 < rest_size:
                dof_distribution[-1][1] += 1
            counter = dof_distribution[-1][1]
            self.counts[p] = (dof_distribution[-1][1] -
                              dof_distribution[-1][0]) * self.M

        assert (counter == self.N)
        self.t_begin, self.t_end = dof_distribution[self.rank]

        # Create a local vector containing the dofs.
        self.X_loc = np.zeros((self.t_end - self.t_begin, self.M),
                              dtype=np.float64)

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
        assert (isinstance(vec_other, VectorTimeMPI))
        assert (vec_other.X_loc.shape == self.X_loc.shape)
        dot_loc = np.dot(self.X_loc.reshape(-1), vec_other.X_loc.reshape(-1))
        dot_glob = self.comm.allreduce(dot_loc)
        return dot_glob
