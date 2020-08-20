import numpy as np

from mpi_vector import KronVectorMPI


def PCG(T, P, b, w0=None, kmax=100000, rtol=1e-5, callback=None):
    def inner(x, y):
        return x.dot(y)

    if w0 is None:
        if isinstance(b, KronVectorMPI):
            w = KronVectorMPI(b.dofs_distr)
        else:
            w = np.zeros(b.shape)
    else:
        w = w0

    iters = 0
    sq_rhs_norm = inner(b, b)
    if sq_rhs_norm == 0: return w, iters

    threshold = rtol * rtol * sq_rhs_norm

    r = b - T @ w
    if inner(r, r) < threshold: return w, iters

    p = P @ r
    abs_r = inner(r, p)
    for k in range(1, kmax):
        iters += 1
        t = T @ p
        alpha = abs_r / inner(p, t)
        w += alpha * p
        r -= alpha * t
        del t
        if callback is not None:
            callback(w, r, k)
        if inner(r, r) < threshold: break
        z = P @ r
        abs_r_old = abs_r
        abs_r = inner(r, z)
        beta = abs_r / abs_r_old
        p *= beta
        p += z
    return w, iters
