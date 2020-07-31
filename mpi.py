from ngsolve import *


# Parallel-in-time only
def setup_local_communicators(X, Y):
    assert X.time.globalorder == 1 and Y.time.globalorder >= 1
    assert X.time.mesh.ne % mpi_world.size == 0
    assert mpi_world.size > 1

    n = mpi_world.rank
    N = mpi_world.size
    slice_size = int(X.time.mesh.ne / N)
    print("hier", n, N)
    mpi_timeslice = mpi_world.SubComm([(n - 1) % N, n, (n + 1) % N])
    print("daar")
    return mpi_timeslice
