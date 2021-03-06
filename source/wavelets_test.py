from math import sqrt

import numpy as np
import scipy
from mpi4py import MPI

from .mpi_kron import as_matrix
from .mpi_vector import DofDistributionMPI
from .wavelets import (TransposedWaveletTransformKronIdentityMPI,
                       WaveletTransformKronIdentityMPI, WaveletTransformMat,
                       WaveletTransformOp)


def test_mat_equals_matfree():
    for J in range(1, 8):
        WOp = WaveletTransformOp(J)
        WMat = WaveletTransformMat(J)
        assert np.allclose(as_matrix(WOp), as_matrix(WMat))
        assert np.allclose(as_matrix(WOp.T), as_matrix(WMat).T)


def test_wavelet_transform_works():
    J = 4
    WOp = WaveletTransformOp(J)

    I = np.eye(2**J + 1)

    # Wavelets level 0.
    assert (np.allclose(WOp @ I[:, 0], np.linspace(1, 0, 2**J + 1)))
    assert (np.allclose(WOp @ I[:, 1], np.linspace(0, 1, 2**J + 1)))

    # Wavelet of level 1.
    y = WOp @ I[:, 2]
    assert (np.allclose(y[:2**(J - 1) + 1],
                        np.linspace(-sqrt(2), sqrt(2), 2**(J - 1) + 1)))
    assert (np.allclose(y[2**(J - 1):],
                        np.linspace(sqrt(2), -sqrt(2), 2**(J - 1) + 1)))

    # Wavelet of level 2.
    y = WOp @ I[:, 3]
    assert (np.allclose(y[0], -2))
    y = WOp @ I[:, 4]
    assert (np.allclose(y[-1], -2))


def test_interleaved_wavelet_transform_works():
    WOpJ2 = WaveletTransformOp(2, interleaved=True)
    assert np.allclose(as_matrix(WOpJ2),
                       [[1, -2, -np.sqrt(2), 0, 0], [3 / 4, 2, 0, 0, 1 / 4],
                        [1 / 2, -1, np.sqrt(2), -1, 1 / 2],
                        [1 / 4, 0, 0, 2, 3 / 4], [0, 0, -np.sqrt(2), -2, 1]])
    J = 4
    WOp = WaveletTransformOp(J, interleaved=True)

    I = np.eye(2**J + 1)

    # Wavelets level 0.
    assert (np.allclose(WOp @ I[:, 0], np.linspace(1, 0, 2**J + 1)))
    assert (np.allclose(WOp @ I[:, -1], np.linspace(0, 1, 2**J + 1)))

    # Wavelet of level 1.
    y = WOp @ I[:, 2**(J - 1)]
    assert (np.allclose(y[:2**(J - 1) + 1],
                        np.linspace(-sqrt(2), sqrt(2), 2**(J - 1) + 1)))
    assert (np.allclose(y[2**(J - 1):],
                        np.linspace(sqrt(2), -sqrt(2), 2**(J - 1) + 1)))

    # Apply all the split operations after eachoter
    Wmat = scipy.sparse.eye(2**J + 1, 2**J + 1, format='csr')
    for j in range(1, J + 1):
        Wmat += WOp.split(j) @ Wmat
    assert np.allclose(as_matrix(WOp), as_matrix(Wmat))


def test_mpi_wavelet_transform_works():
    J = 4
    N = 2**J + 1
    M = 1
    dofs_distr = DofDistributionMPI(MPI.COMM_WORLD, N, M)
    rank = MPI.COMM_WORLD.Get_rank()
    WOp = WaveletTransformKronIdentityMPI(dofs_distr, J)
    WOpT = TransposedWaveletTransformKronIdentityMPI(dofs_distr, J)
    WOp2 = WaveletTransformOp(J, interleaved=True)
    WOpmat = WOp.as_global_matrix()
    WOpTmat = WOpT.as_global_matrix()
    if rank == 0:
        assert np.allclose(WOpmat, as_matrix(WOp2))
        assert np.allclose(WOpTmat, WOpmat.T)
