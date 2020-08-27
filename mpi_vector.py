import numpy as np

from mpi4py import MPI


class DofDistributionMPI:
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

        self.dof2proc = np.zeros(self.N)
        for p, (t_begin, t_end) in enumerate(self.dof_distribution):
            self.dof2proc[t_begin:t_end] = p


class KronVectorMPI:
    """ This class represents a vector that is parallized in the first
    component (e.g., time).
    """
    def __init__(self, dofs_distr, initial_data=None):
        self.dofs_distr = dofs_distr

        # Convenience
        self.t_begin = dofs_distr.t_begin
        self.t_end = dofs_distr.t_end
        self.N = dofs_distr.N
        self.M = dofs_distr.M
        self.rank = dofs_distr.rank

        # Initialize to zero.
        self.reset(initial_data)

    @property
    def X_loc(self):
        return self._X_loc

    def reset(self, initial_data=None):
        self.communicated_bdr = False
        self.X_loc_bdr = None

        if initial_data is None:
            self._X_loc = np.zeros((self.t_end - self.t_begin, self.M),
                                   dtype=np.float64)
        else:
            assert initial_data.shape == (self.t_end - self.t_begin, self.M)
            self._X_loc = initial_data.copy()

    def copy(self):
        cpy = KronVectorMPI(self.dofs_distr, self.X_loc)
        return cpy

    def _invalidate(self):
        """ Invalides the bdr communicated dofs. """
        if self.communicated_bdr:
            self.communicated_bdr = False
            self.X_loc_bdr.setflags(write=True)
            self.X_loc.setflags(write=True)

    def __iadd__(self, other):
        self._invalidate()
        self._X_loc += other.X_loc
        return self

    def __add__(self, other):
        vec_out = self.copy()
        vec_out += other
        return vec_out

    def __isub__(self, other):
        self._invalidate()
        self._X_loc -= other.X_loc
        return self

    def __sub__(self, other):
        vec_out = self.copy()
        vec_out -= other
        return vec_out

    def __imul__(self, other):
        self._invalidate()
        self._X_loc *= other
        return self

    def __rmul__(self, other):
        vec_out = self.copy()
        vec_out *= other
        return vec_out

    def __itruediv__(self, other):
        self._invalidate()
        self._X_loc /= other
        return self

    def __truediv__(self, other):
        vec_out = self.copy()
        vec_out /= other
        return vec_out

    def scatter(self, X_glob):
        data = [None, None, None, MPI.DOUBLE]
        if self.rank == 0:
            data = [
                X_glob, self.dofs_distr.counts, self.dofs_distr.displs,
                MPI.DOUBLE
            ]

        self._invalidate()
        self.dofs_distr.comm.Scatterv(data, self.X_loc)

    def gather(self, X_glob):
        self.dofs_distr.comm.Gatherv(self.X_loc, [
            X_glob, self.dofs_distr.counts, self.dofs_distr.displs, MPI.DOUBLE
        ])

    def communicate_bdr(self, callback=None):
        """ Communicates the bdr dofs, and calls the callback function
            for computaitons that do not require the bdr dofs."""
        if self.communicated_bdr:
            callback()
            return 0.0

        # Create a local vector containing an enlarged number of dofs.
        if self.X_loc_bdr is None:
            self.X_loc_bdr = np.zeros(
                (self.dofs_distr.t_end - self.dofs_distr.t_begin + 2, self.M),
                dtype=np.float64)

        # Communicate.
        reqs = []
        if self.rank > 0:
            reqs.append(
                self.dofs_distr.comm.Isend(self.X_loc[0, :], self.rank - 1))
            reqs.append(
                self.dofs_distr.comm.Irecv(self.X_loc_bdr[0, :],
                                           source=self.rank - 1))

        if self.rank + 1 < self.dofs_distr.size:
            reqs.append(
                self.dofs_distr.comm.Isend(self.X_loc[-1, :], self.rank + 1))
            reqs.append(
                self.dofs_distr.comm.Irecv(self.X_loc_bdr[-1, :],
                                           source=self.rank + 1))

        # Copy input.
        self.X_loc_bdr[1:-1] = self.X_loc

        # Make X_loc a view.
        self._X_loc = self.X_loc_bdr[1:-1]
        self._X_loc.setflags(write=False)
        self.X_loc_bdr.setflags(write=False)

        # Do computation that doesn't require the bdr to be present.
        if callback is not None:
            callback()

        # Wait for data communication to complete.
        start_time = MPI.Wtime()
        MPI.Request.Waitall(reqs)
        time_communication = MPI.Wtime() - start_time

        self.communicated_bdr = True
        return time_communication

    def communicate_dofs(self, comm_dofs):
        X_recv = {recv: np.zeros(self.M) for (send, recv) in comm_dofs}
        reqs = []
        for send, recv in comm_dofs:
            reqs.append(
                self.dofs_distr.comm.Isend(self.X_loc[send - self.t_begin, :],
                                           dest=self.dofs_distr.dof2proc[recv],
                                           tag=send))
            reqs.append(
                self.dofs_distr.comm.Irecv(
                    X_recv[recv],
                    source=self.dofs_distr.dof2proc[recv],
                    tag=recv))

        return X_recv, reqs

    def dot(self, vec_other):
        assert (isinstance(vec_other, KronVectorMPI))
        assert (vec_other.X_loc.shape == self.X_loc.shape)
        dot_loc = np.dot(self.X_loc.reshape(-1), vec_other.X_loc.reshape(-1))
        dot_glob = self.dofs_distr.comm.allreduce(dot_loc)
        return dot_glob

    def permute(self, vec_perm=None):
        """ Permutes the order of the kronecker product. """
        start_time = MPI.Wtime()
        if vec_perm is None:
            vec_perm = KronVectorMPI(
                DofDistributionMPI(self.dofs_distr.comm, self.M, self.N))
        else:
            assert (vec_perm.N == self.M and vec_perm.M == self.N)

        # TODO: Do this using scatter.

        # Now lets send all the dofs.
        X_loc_T = self.X_loc.T.copy()
        reqs = []
        for p in range(self.dofs_distr.size):
            x_begin, x_end = vec_perm.dofs_distr.dof_distribution[p]
            reqs.append(
                self.dofs_distr.comm.Isend(X_loc_T[x_begin:x_end, :], dest=p))

        # Receive all the dofs.
        for p in range(self.dofs_distr.size):
            t_begin, t_end = self.dofs_distr.dof_distribution[p]
            x_begin, x_end = vec_perm.t_begin, vec_perm.t_end
            buf = np.empty((x_end - x_begin, t_end - t_begin))
            self.dofs_distr.comm.Recv(buf, source=p)
            vec_perm.X_loc[:, t_begin:t_end] = buf

        MPI.Request.Waitall(reqs)
        return vec_perm, MPI.Wtime() - start_time
