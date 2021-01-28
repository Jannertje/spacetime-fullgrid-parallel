import numpy as np
from .mpi_vector import KronVectorMPI


def PCG(T, P, b, w0=None, kmax=100000, eps=1e-6, callback=None):
    """ Preconditioned Conjuage Gradients with algebraic stopping criterium. """
    if w0 is None:
        if isinstance(b, KronVectorMPI):
            w = KronVectorMPI(b.dofs_distr)
        else:
            w = np.zeros(b.shape)
    else:
        w = w0

    iters = 0
    sq_rhs_norm = b.dot(b)
    if sq_rhs_norm == 0: return w, iters

    r = b - T @ w

    p = P @ r
    abs_r = r.dot(p)
    if abs_r < eps * eps: return w, iters
    for k in range(1, kmax):
        iters += 1
        t = T @ p
        alpha = abs_r / p.dot(t)
        w += alpha * p
        r -= alpha * t
        del t
        if callback is not None:
            callback(w, r, k)
        z = P @ r
        abs_r_old = abs_r
        abs_r = r.dot(z)
        if abs_r < eps * eps: break
        beta = abs_r / abs_r_old
        p *= beta
        p += z
        del z
    return w, iters
