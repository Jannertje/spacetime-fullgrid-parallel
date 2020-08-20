import sys

import numpy as np
from mpi4py import MPI

from mpi_vector import KronVectorMPI, DofDistributionMPI


def test_dot():
    N = 9
    M = 13
    comm = MPI.COMM_WORLD
    dofs_distr = DofDistributionMPI(comm, N, M)
    vec = KronVectorMPI(dofs_distr)
    x_glob = None
    if dofs_distr.rank == 0:
        x_glob = np.arange(0, N * M) * 1.0
    vec.scatter(x_glob)
    norm_vec_sqr = vec.dot(vec)
    if dofs_distr.rank == 0:
        assert (np.allclose(norm_vec_sqr, np.dot(x_glob, x_glob)))

    x_glob_2 = None
    vec_2 = KronVectorMPI(dofs_distr)
    if dofs_distr.rank == 0:
        x_glob_2 = np.random.rand(N * M)
    vec_2.scatter(x_glob_2)
    ip_vec_vec2 = vec.dot(vec_2)
    if dofs_distr.rank == 0:
        assert (np.allclose(ip_vec_vec2, np.dot(x_glob, x_glob_2)))


def test_permute():
    for N in range(4, 25):
        M = 244
        comm = MPI.COMM_WORLD
        dofs_distr = DofDistributionMPI(comm, N, M)
        vec = KronVectorMPI(dofs_distr)
        t_glob = None
        x_glob = None
        if dofs_distr.rank == 0:
            x_glob = np.empty(N * M, dtype=np.float64)
            t_glob = np.arange(0, N * M) * 1.0
        vec.scatter(t_glob)
        vec_space, _ = vec.permute()
        comm.Barrier()

        vec_space.gather(x_glob)
        if dofs_distr.rank == 0:
            assert (np.allclose(t_glob.reshape(N, M), x_glob.reshape(M, N).T))
