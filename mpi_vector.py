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
        self.dof_distribution = []
        self.dof2proc = np.zeros(self.N)
        counter = 0
        for p in range(self.size):
            self.dof_distribution.append([counter, counter + block_size])
            if self.size - p - 1 < rest_size:
                self.dof_distribution[-1][1] += 1
            counter = self.dof_distribution[-1][1]

            for t in range(*self.dof_distribution[-1]):
                self.dof2proc[t] = p

        assert (counter == self.N)

        # Set MPI counts/displs
        self.counts = [(t_end - t_begin) * self.M
                       for (t_begin, t_end) in self.dof_distribution]
        self.displs = np.zeros(self.size)
        for p in range(1, self.size):
            self.displs[p] = self.displs[p - 1] + self.counts[p - 1]

        self.t_begin, self.t_end = self.dof_distribution[self.rank]

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


comm = MPI.COMM_WORLD
vec = VectorTimeMPI(comm, 9, 3)
X_glob = None
if comm.Get_rank() == 0:
    X_glob = np.arange(0, vec.N * vec.M) * 1.0
vec.scatter(X_glob)
vec.X_loc *= 2
if comm.Get_rank() == 0:
    X_glob = np.zeros(vec.N * vec.M) * 1.0
vec.gather(X_glob)
if comm.Get_rank() == 0:
    print(X_glob.reshape(9, 3))

comm.Barrier()
print(vec.rank, vec.communicate_bdr())
