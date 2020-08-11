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

    # Check that wavelettransform is identity for wavelets on the toplevel.
    I = np.eye(2**J + 1)
    N_coarse = 2**(J - 1) + 1
    for n, wavelet in enumerate(WOp.qT[J]):
        y = WOp @ as_matrix(wavelet).reshape(-1)
        assert (np.allclose(y, I[:, N_coarse + n]))
