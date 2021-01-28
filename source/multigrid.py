import os

import numpy as np
import psutil
import scipy.sparse
from mpi4py import MPI
from petsc4py import PETSc
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator, splu

from .lanczos import Lanczos
from .linop import AsLinearOperator
from .mpi_kron import as_matrix
from .mpi_shared_mem import shared_sparse_matrix


class MeshHierarchy:
    """ Builds a mesh hierarchy, and prolongation/restriction matrices. """
    def __init__(self, fes, shared_comm=None):
        J = None
        if shared_comm is None or shared_comm.rank == 0:
            mesh = fes.mesh
            parents = np.array(
                [mesh.GetParentVertices(v) for v in range(mesh.nv)])
            levels = np.zeros(len(parents), dtype=np.int)
            for v in reversed(range(mesh.nv)):
                for gp in parents[v]:
                    if gp == -1: break
                    levels[gp] = levels[v] - 1
            levels = levels - levels[0]
            J = levels[-1]
            nverts = np.zeros(J + 1, dtype=np.int)
            for v in levels:
                nverts[v] += 1
            nverts = np.cumsum(nverts, dtype=np.int)
            fdofs = np.array(fes.FreeDofs())
            I = scipy.sparse.eye(len(fdofs)).tocsr()

            # Maps a full matrix to the restricted variant.
            self.F_J = I[fdofs, :]

            P_mats = []
            for j in range(J):
                nc = nverts[j]
                nf = nverts[j + 1]
                row, col, val = np.empty(
                    nc + 2 * (nf - nc), dtype=np.int), np.empty(
                        nc + 2 * (nf - nc),
                        dtype=np.int), np.empty(nc + 2 * (nf - nc))

                # Identity along the diagonal.
                row[0:nc] = np.arange(nc)
                col[0:nc] = np.arange(nc)
                val[0:nc] = 1

                row[nc:] = np.repeat(np.arange(nc, nf), repeats=2)
                col[nc:] = parents[nc:nf].reshape(-1)
                val[nc:] = 0.5

                # Now filter out all the vertices that do not correspond to dofs.
                P_mats.append(I[fdofs[:nf], :nf] @ csr_matrix(
                    (val, (row, col)), shape=(nf, nc)) @ I[:nc, fdofs[:nc]])
                R_mats = [P.T.tocsr() for P in P_mats]

        # set variables
        self.shared_comm = shared_comm
        if shared_comm is None:
            self.J = J
            self.P_mats = P_mats
            self.R_mats = R_mats
        else:
            self.J = shared_comm.bcast(J)
            # Create shared matrices.
            if shared_comm.rank != 0:
                P_mats = [None for _ in range(self.J)]
                R_mats = [None for _ in range(self.J)]

            self.P_mats = [
                shared_sparse_matrix(P, shared_comm) for P in P_mats
            ]
            self.R_mats = [
                shared_sparse_matrix(R, shared_comm) for R in R_mats
            ]


class Smoother:
    """ SOR smoother, in python. """
    def __init__(self, mat):
        self.mat_rows = [row for row in mat]
        self.invdiag = mat.diagonal()**-1

    def PreSmooth(self, u, f):
        for i, row in enumerate(self.mat_rows):
            ax = row @ u
            u[i] += self.invdiag[i] * (f[i] - ax)

    def PostSmooth(self, u, f):
        for i, row in reversed(list(enumerate(self.mat_rows))):
            ax = row @ u
            u[i] += self.invdiag[i] * (f[i] - ax)


class PETScSMoother:
    """ Uses PETSc for SOR smoothing. Faster implementation in C++. """
    def __init__(self, mat, its):
        self.its = its
        self.mat_petsc = PETSc.Mat().createAIJWithArrays(size=mat.shape,
                                                         csr=(mat.indptr,
                                                              mat.indices,
                                                              mat.data),
                                                         comm=PETSc.COMM_SELF)
        self.f = np.empty(mat.shape[1])
        self.f_petsc = PETSc.Vec().createWithArray(self.f,
                                                   comm=PETSc.COMM_SELF)

    def PreSmooth(self, u, f):
        self.f_petsc.setArray(f)
        u_petsc = PETSc.Vec().createWithArray(u, comm=PETSc.COMM_SELF)
        self.mat_petsc.SOR(self.f_petsc,
                           u_petsc,
                           its=self.its,
                           sortype=self.mat_petsc.SORType.FORWARD_SWEEP)

    def PostSmooth(self, u, f):
        self.f_petsc.setArray(f)
        u_petsc = PETSc.Vec().createWithArray(u, comm=PETSc.COMM_SELF)
        self.mat_petsc.SOR(self.f_petsc,
                           u_petsc,
                           its=self.its,
                           sortype=self.mat_petsc.SORType.BACKWARD_SWEEP)


class MultiGrid(LinearOperator):
    """ Simple multigrid implementation for uniform meshes. """
    def __init__(self, mat, hierarchy, smoothsteps=2, vcycles=1):
        self.num_applies = 0
        self.time_applies = 0
        self.hierarchy = hierarchy
        self.smoothsteps = smoothsteps
        self.vcycles = vcycles
        shared_comm = hierarchy.shared_comm

        if shared_comm is None or shared_comm.rank == 0:
            # Store the matrix/smoothers on all levels.
            mats = [mat]
            for j in reversed(range(hierarchy.J)):
                mats = [hierarchy.R_mats[j] @ mats[0] @ hierarchy.P_mats[j]
                        ] + mats

        if shared_comm is None:
            self.mats = mats
        else:
            if shared_comm.rank != 0:
                mats = [None for _ in range(hierarchy.J + 1)]
            self.mats = [
                shared_sparse_matrix(mat, shared_comm) for mat in mats[:-1]
            ] + [mat]

        self.smoothers = [None]
        for j in range(1, hierarchy.J + 1):
            self.smoothers.append(PETScSMoother(self.mats[j], its=smoothsteps))

        # Solve on the coarsest mesh
        self.coarse_solver = splu(
            self.mats[0].T,
            options={"SymmetricMode": True},
            permc_spec="MMD_AT_PLUS_A",
        )
        super().__init__(shape=self.mats[-1].shape, dtype=np.float64)

    def MGM(self, j, u_j, f_j):
        if j == 0:
            u_j[:] = self.coarse_solver.solve(f_j)
        else:
            self.smoothers[j].PreSmooth(u_j, f_j)

            d_cj = self.hierarchy.R_mats[j - 1] @ (self.mats[j] @ u_j - f_j)

            # Recurse.
            u_cj = np.zeros(self.mats[j - 1].shape[0])
            self.MGM(j - 1, u_cj, d_cj)

            u_j -= self.hierarchy.P_mats[j - 1] @ u_cj

            self.smoothers[j].PostSmooth(u_j, f_j)

    def _matvec(self, b):
        start_time = MPI.Wtime()

        x = np.zeros(b.shape[0])
        for _ in range(self.vcycles):
            self.MGM(self.hierarchy.J, x, b.reshape(-1))

        self.num_applies += 1
        self.time_applies += MPI.Wtime() - start_time
        return x

    def time_per_apply(self):
        assert (self.time_applies)
        return self.time_applies / self.num_applies


if __name__ == "__main__":
    shared_comm = MPI.COMM_WORLD.Split_type(MPI.COMM_TYPE_SHARED)
    if shared_comm.rank == 0:
        from mesh import construct_2d_square_mesh
        from ngsolve import (H1, BaseMatrix, BilinearForm, InnerProduct,
                             Preconditioner, TaskManager, ds, dx, grad,
                             ngsglobals)
        mesh, bc = construct_2d_square_mesh(9)
        fes = H1(mesh, order=1, dirichlet=bc)
        A_bf = BilForm(
            fes,
            bilform_lambda=lambda u, v: InnerProduct(grad(u), grad(v)) * dx)
        A = A_bf.assemble()
    else:
        A = None
        fes = None
    hierarch = MeshHierarchy(fes, shared_comm=shared_comm)
    A = shared_sparse_matrix(A, shared_comm)
    print(A.shape)

    for smoothsteps in [1, 2, 3, 4]:
        for vcycles in [1, 2, 3, 4]:
            mg = MultiGrid(A,
                           hierarch,
                           smoothsteps=smoothsteps,
                           vcycles=vcycles)
            lanczos = Lanczos(A, mg)
            print('smoothsteps={} vcycles={} time_per_apply={} lanczos:{}'.
                  format(smoothsteps, vcycles, mg.time_per_apply(), lanczos))
