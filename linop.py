from ngsolve import *
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator
import numpy as np


def KronLinOp(mat_time, mat_space):
    N, K = mat_time.shape
    M, L = mat_space.shape

    def matvec(x):
        X = x.reshape(K, L)
        return mat_space.dot(mat_time.dot(X).T).T.reshape(-1)

    return sp.linalg.LinearOperator(matvec=matvec, shape=(N * M, K * L))


def InvLinOp(mat):
    splu = sp.linalg.splu(
        mat,
        options={"SymmetricMode": True},
        permc_spec="MMD_AT_PLUS_A",
    )
    return sp.linalg.LinearOperator(matvec=lambda x: splu.solve(x),
                                    shape=mat.shape)


def BlockDiagLinOp(linops):
    height = sum(linop.shape[0] for linop in linops)
    width = sum(linop.shape[1] for linop in linops)

    def matvec(x):
        y = np.zeros(height, dtype=x.dtype)
        start = 0
        for linop in linops:
            end = start + linop.shape[0]
            y[start:end] += linop.dot(x[start:end])
            start = end
        return y

    return LinearOperator(matvec=matvec, shape=(height, width))


def BlockLinOp(linops):
    height = sum(row[0].shape[0] for row in linops)
    width = sum(mat.shape[1] for mat in linops[0])

    def matvec(x):
        y = np.zeros(height, dtype=x.dtype)
        row_start = col_start = 0
        for row in linops:
            row_end = row_start + row[0].shape[0]
            for linop in row:
                col_end = col_start + linop.shape[1]
                y[row_start:row_end] += linop.dot(x[col_start:col_end])
                col_start = col_end
            row_start = row_end
            col_start = 0
        return y

    return LinearOperator(matvec=matvec, shape=(height, width))


def CompositeLinOp(linops):
    def matvec(x):
        y = x
        for linop in reversed(linops):
            y = linop.dot(y)
        return y

    return LinearOperator(matvec=matvec,
                          shape=(linops[0].shape[0], linops[-1].shape[1]))


def AsLinearOperator(ngmat, freedofs):
    tmp1 = ngmat.CreateRowVector()
    tmp2 = ngmat.CreateColVector()

    def step(v):
        tmp1.FV().NumPy()[freedofs] = v.reshape(-1)
        tmp2.data = ngmat * tmp1
        return tmp2.FV().NumPy()[freedofs].reshape(v.shape)

    return LinearOperator((len(freedofs), len(freedofs)), step)
