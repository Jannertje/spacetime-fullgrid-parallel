import numpy as np

from mpi_vector import VectorTimeMPI


def PCG(T, P, b, w0=None, kmax=100000, rtol=1e-5, callback=None):
    def inner(x, y):
        return x.dot(y)

    if w0 is None:
        if isinstance(b, VectorTimeMPI):
            w0 = VectorTimeMPI(b.comm, b.dofs_time, b.dofs_space)
        else:
            w0 = np.zeros(b.shape)

    sq_rhs_norm = inner(b, b)
    if sq_rhs_norm == 0: return w0

    threshold = rtol * rtol * sq_rhs_norm
    r = b - T @ w0
    if inner(r, r) < threshold: return w0

    w = w0
    p = P @ r
    abs_r = inner(r, p)
    for k in range(1, kmax):
        t = T @ p
        alpha = abs_r / inner(p, t)
        w += alpha * p
        r -= alpha * t
        if callback is not None:
            callback(w, r, k)
        if inner(r, r) < threshold: break
        z = P @ r
        abs_r_old = abs_r
        abs_r = inner(r, z)
        beta = abs_r / abs_r_old
        p = z + beta * p
    return w
