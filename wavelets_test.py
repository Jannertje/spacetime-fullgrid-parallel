from math import sqrt

import numpy as np

from mpi_kron import as_matrix
from wavelets import WaveletTransformMat, WaveletTransformOp


def test_mat_equals_matfree():
    for J in range(1, 8):
        WOp = WaveletTransformOp(J)
        WMat = WaveletTransformMat(J)
        assert (np.allclose(as_matrix(WOp), as_matrix(WMat)))
        assert (np.allclose(as_matrix(WOp.H), as_matrix(WMat).T))


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
