from mpi4py import MPI
import scipy.sparse
import numpy as np
import time
import sys


def shared_numpy_array(array, shared_comm, dtype=np.float64):
    dtype = np.dtype(dtype)
    shape = None
    nbytes = 0
    if shared_comm.rank == 0:
        assert (isinstance(array, np.ndarray))
        assert (array.dtype == dtype)
        nbytes = array.nbytes
        shape = array.shape
    shape = shared_comm.bcast(shape)
    win = MPI.Win.Allocate_shared(nbytes, dtype.itemsize, comm=shared_comm)
    buf, _ = win.Shared_query(0)
    shared_array = np.ndarray(buffer=buf, dtype=dtype, shape=shape)

    if shared_comm.rank == 0:
        shared_array[:] = array
    win.Fence()
    #shared_array.setflags(write=False)
    return shared_array


def shared_sparse_matrix(mat, shared_comm):
    data, indices, indptr = None, None, None
    shape = None
    if shared_comm.rank == 0:
        assert (scipy.sparse.isspmatrix_csr(mat))
        data = mat.data
        indices = mat.indices
        indptr = mat.indptr
        shape = mat.shape
    shape = shared_comm.bcast(shape)

    shared_data = shared_numpy_array(data, shared_comm)
    shared_indices = shared_numpy_array(indices, shared_comm, np.int32)
    shared_indptr = shared_numpy_array(indptr, shared_comm, np.int32)
    return scipy.sparse.csr_matrix(
        (shared_data, shared_indices, shared_indptr), shape=shape)


#shared_comm = MPI.COMM_WORLD.Split_type(MPI.COMM_TYPE_SHARED)
#
#mat = None
#if shared_comm.rank == 0:
#    mat = np.array([[3.5, 13., 28.5, 50.,
#                     77.5], [-5., -23., -53., -95., -149.],
#                    [2.5, 11., 25.5, 46., 72.5]])
#    mat = scipy.sparse.spdiags(mat, (1, 0, -1), 5, 5).tocsr()
#
#shared_mat = shared_sparse_matrix(mat, shared_comm)
#
#print(shared_comm.rank, shared_mat)
#
#if shared_comm.rank == 1:
#    shared_mat[0, 0] = 1337
#shared_comm.Barrier()
#print(shared_comm.rank, shared_mat[0, 0])
