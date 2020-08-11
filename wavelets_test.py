import numpy as np

from mpi_kron import as_matrix
from wavelets import WaveletTransformMat, WaveletTransformOp


def test_mat_equals_matfree():
    for J in range(1, 8):
        WOp = WaveletTransformOp(J)
        WMat = WaveletTransformMat(J)
        assert (np.allclose(as_matrix(WOp), as_matrix(WMat)))
        assert (np.allclose(as_matrix(WOp.H), as_matrix(WMat).T))
