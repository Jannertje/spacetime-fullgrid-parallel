import numpy as np
from scipy.sparse.linalg import splu, LinearOperator
from petsc4py import PETSc
import scipy.sparse
from netgen.csg import unit_cube
from netgen.geom2d import unit_square
from ngsolve import (H1, BaseMatrix, BilinearForm, InnerProduct,
                     Preconditioner, TaskManager, ds, dx, grad, ngsglobals)
from ngsolve.la import EigenValues_Preconditioner
from scipy.sparse import csr_matrix

from bilform import BilForm
from lanczos import Lanczos
from linop import AsLinearOperator
from mesh import construct_2d_square_mesh
from mpi_kron import as_matrix


class MeshHierarchy:
    def __init__(self, fes):
        mesh = fes.mesh
        parents = np.array([mesh.GetParentVertices(v) for v in range(mesh.nv)])
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

        # set variables
        self.J = J
        self.P_mats = P_mats
        self.R_mats = [P.T.tocsr() for P in P_mats]


class Smoother:
    def __init__(self, mat):
        self.mat_rows = [row for row in mat]
        self.invdiag = mat.diagonal()**-1

    def PreSmooth(self, u, f):
        for i, row in enumerate(self.mat_rows):
            ax = row @ u
            u[i] += self.invdiag[i] * (f[i] - ax)
        return u

    def PostSmooth(self, u, f):
        for i, row in reversed(list(enumerate(self.mat_rows))):
            ax = row @ u
            u[i] += self.invdiag[i] * (f[i] - ax)
        return u


class PETScSMoother:
    def __init__(self, mat):
        self.mat_petsc = PETSc.Mat().createAIJWithArrays(size=mat.shape,
                                                         csr=(mat.indptr,
                                                              mat.indices,
                                                              mat.data))

    def PreSmooth(self, u, f):
        f_petsc = PETSc.Vec().createWithArray(f)
        u_petsc = PETSc.Vec().createWithArray(u)
        self.mat_petsc.SOR(f_petsc,
                           u_petsc,
                           sortype=self.mat_petsc.SORType.FORWARD_SWEEP)
        return u

    def PostSmooth(self, u, f):
        f_petsc = PETSc.Vec().createWithArray(f)
        u_petsc = PETSc.Vec().createWithArray(u)
        self.mat_petsc.SOR(f_petsc,
                           u_petsc,
                           sortype=self.mat_petsc.SORType.BACKWARD_SWEEP)
        return u


class MultiGrid(LinearOperator):
    def __init__(self, mat, hierarchy, smoothsteps=1, vcycles=1):
        super().__init__(shape=mat.shape, dtype=np.float64)
        # Store the matrix/smoothers on all levels.
        self.mats = [mat]
        self.hierarchy = hierarchy
        for j in reversed(range(hierarchy.J)):
            self.mats = [
                hierarchy.R_mats[j] @ self.mats[0] @ hierarchy.P_mats[j]
            ] + self.mats

        self.smoothers = [None]
        for j in range(1, hierarchy.J + 1):
            self.smoothers.append(PETScSMoother(self.mats[j]))

        # Solve on the coarsest mesh
        self.coarse_solver = splu(
            self.mats[0].T,
            options={"SymmetricMode": True},
            permc_spec="MMD_AT_PLUS_A",
        )

        self.smoothsteps = smoothsteps
        self.vcycles = vcycles

    def MGM(self, j, u_j, f_j):
        if j == 0:
            u_j = self.coarse_solver.solve(f_j)
        else:
            for _ in range(self.smoothsteps):
                self.smoothers[j].PreSmooth(u_j, f_j)

            d_cj = self.hierarchy.R_mats[j - 1] @ (self.mats[j] @ u_j - f_j)
            u_cj = self.MGM(j - 1, np.zeros(self.mats[j - 1].shape[0]), d_cj)
            u_j -= self.hierarchy.P_mats[j - 1] @ u_cj

            for _ in range(self.smoothsteps):
                self.smoothers[j].PostSmooth(u_j, f_j)

        return u_j

    def _matvec(self, b):
        x = np.zeros(b.shape[0])
        for _ in range(self.vcycles):
            self.MGM(self.hierarchy.J, x, b.reshape(-1))
        return x


if __name__ == "__main__":
    mesh, bc = construct_2d_square_mesh(1)
    fes = H1(mesh, order=1, dirichlet=bc)
    A_bf = BilForm(
        fes, bilform_lambda=lambda u, v: InnerProduct(grad(u), grad(v)) * dx)
    A = A_bf.assemble()
    A_petsc = PETSc.Mat().createAIJWithArrays(size=A.shape,
                                              csr=(A.indptr, A.indices,
                                                   A.data))
    hierarch = MeshHierarchy(fes)

    for smoothsteps in [1]:
        for vcycles in [1]:
            mg = MultiGrid(A,
                           hierarch,
                           smoothsteps=smoothsteps,
                           vcycles=vcycles)
            print('smoothsteps={} vcycles={} lanczos:{}'.format(
                smoothsteps, vcycles, Lanczos(A, mg)))
