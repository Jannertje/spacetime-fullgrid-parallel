import numpy as np
from mpi4py import MPI

from mpi_vector import KronVectorMPI


def test_dot():
    N = 9
    M = 13
    comm = MPI.COMM_WORLD
    vec = KronVectorMPI(comm, N, M)
    x_glob = None
    if vec.rank == 0:
        x_glob = np.arange(0, N * M) * 1.0
    vec.scatter(x_glob)
    norm_vec_sqr = vec.dot(vec)
    if vec.rank == 0:
        assert (np.allclose(norm_vec_sqr, np.dot(x_glob, x_glob)))

    x_glob_2 = None
    vec_2 = KronVectorMPI(comm, N, M)
    if vec.rank == 0:
        x_glob_2 = np.random.rand(N * M)
    vec_2.scatter(x_glob_2)
    ip_vec_vec2 = vec.dot(vec_2)
    if vec.rank == 0:
        assert (np.allclose(ip_vec_vec2, np.dot(x_glob, x_glob_2)))


def test_permute():
    N = 9
    M = 13
    comm = MPI.COMM_WORLD
    vec = KronVectorMPI(comm, N, M)
    t_glob = None
    x_glob = None
    if vec.rank == 0:
        x_glob = np.empty(N * M, dtype=np.float64)
        t_glob = np.arange(0, N * M) * 1.0
    vec.scatter(t_glob)
    vec_space = vec.permute()

    vec_space.gather(x_glob)
    if vec.rank == 0:
        assert (np.allclose(t_glob.reshape(N, M), x_glob.reshape(M, N).T))
