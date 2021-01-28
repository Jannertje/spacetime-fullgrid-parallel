import numpy as np
from mpi4py import MPI
from netgen.csg import unit_cube
from netgen.geom2d import unit_square
from ngsolve import (H1, BaseMatrix, BilinearForm, InnerProduct,
                     Preconditioner, TaskManager, ds, dx, grad, ngsglobals)
from ngsolve.la import EigenValues_Preconditioner
from scipy.sparse import csr_matrix

from .lanczos import Lanczos
from .linop import AsLinearOperator
from .mesh import construct_2d_square_mesh, construct_3d_cube_mesh
from .mpi_kron import as_matrix
from .multigrid import MeshHierarchy, MultiGrid, PETScSMoother, Smoother
from .ngsolve_helper import BilForm

ngsglobals.msg_level = 0


def test_prolongation():
    for meshfn in [construct_2d_square_mesh, construct_3d_cube_mesh]:
        mesh_2, bc = meshfn(2)
        fes_2 = H1(mesh_2, order=1, dirichlet=bc)
        A_2_bf = BilForm(
            fes_2,
            bilform_lambda=lambda u, v: InnerProduct(grad(u), grad(v)) * dx)
        A_2 = A_2_bf.assemble()

        A_1 = BilForm(H1(meshfn(1)[0], order=1, dirichlet="default"),
                      bilform_lambda=lambda u, v: InnerProduct(
                          grad(u), grad(v)) * dx).assemble()

        A_0 = BilForm(H1(meshfn(0)[0], order=1, dirichlet="default"),
                      bilform_lambda=lambda u, v: InnerProduct(
                          grad(u), grad(v)) * dx).assemble()

        hierarch = MeshHierarchy(fes_2)
        A_mats = [A_0, A_1, A_2]
        for j in range(hierarch.J):
            assert np.allclose(
                as_matrix(
                    hierarch.R_mats[j] @ A_mats[j + 1] @ hierarch.P_mats[j]),
                as_matrix(A_mats[j]))


def test_smoother():
    mesh, bc = construct_2d_square_mesh(2)
    fes = H1(mesh, order=1, dirichlet=bc)
    A = BilForm(fes,
                bilform_lambda=lambda u, v: InnerProduct(grad(u), grad(v)) * dx
                ).assemble()
    for smoother in [Smoother(A)]:

        x = np.random.rand(A.shape[1])
        y = A @ x

        x_pre = np.zeros(A.shape[1])
        x_post = np.zeros(A.shape[1])
        for _ in range(150):
            smoother.PreSmooth(x_pre, y)
            smoother.PostSmooth(x_post, y)

        assert np.allclose(x_post, x)
        assert np.allclose(x_pre, x)


def test_multigrid_coarse():
    mesh, bc = construct_2d_square_mesh(0)
    fes = H1(mesh, order=1, dirichlet=bc)
    A = BilForm(fes,
                bilform_lambda=lambda u, v: InnerProduct(grad(u), grad(v)) * dx
                ).assemble()
    hierarchy = MeshHierarchy(fes)
    mg = MultiGrid(A, hierarchy)

    lz = Lanczos(A, mg)
    assert (np.allclose(lz.cond(), 1))


def test_multigrid_smoothingsteps():
    for refines in range(3):
        mesh, bc = construct_2d_square_mesh(refines)
        fes = H1(mesh, order=1, dirichlet=bc)
        A = BilForm(fes,
                    bilform_lambda=lambda u, v: InnerProduct(grad(u), grad(v))
                    * dx).assemble()
        hierarchy = MeshHierarchy(fes)
        mg = MultiGrid(A, hierarchy, smoothsteps=10)
        lz = Lanczos(A, mg)
        assert abs(lz.cond() - 1) < 0.01


def test_multigrid_symmetric():
    for refines in range(4):
        mesh, bc = construct_2d_square_mesh(refines)
        fes = H1(mesh, order=1, dirichlet=bc)
        A = BilForm(fes,
                    bilform_lambda=lambda u, v: InnerProduct(grad(u), grad(v))
                    * dx).assemble()
        hierarchy = MeshHierarchy(fes)
        mg = MultiGrid(A, hierarchy)

        mg_mat = as_matrix(mg)
        assert np.allclose(mg_mat.T, mg_mat)


def test_multigrid_MPI():
    shared_comm = MPI.COMM_WORLD.Split_type(MPI.COMM_TYPE_SHARED)
    for refines in range(4):
        mesh, bc = construct_2d_square_mesh(refines)
        fes = H1(mesh, order=1, dirichlet=bc)
        A = BilForm(fes,
                    bilform_lambda=lambda u, v: InnerProduct(grad(u), grad(v))
                    * dx).assemble()
        hierarchy = MeshHierarchy(fes)
        hierarchy_mpi = MeshHierarchy(fes, shared_comm=shared_comm)
        mg_seq = MultiGrid(A, hierarchy)
        mg_mpi = MultiGrid(A, hierarchy_mpi)
        assert np.allclose(as_matrix(mg_seq), as_matrix(mg_mpi))
