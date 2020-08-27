import numpy as np
import scipy.sparse

from demo import demo
from lanczos import Lanczos
from linalg import PCG
from linop import AsLinearOperator, CompositeLinOp
from mpi4py import MPI
from mpi_heateq import HeatEquationMPI
from mpi_kron import (BlockDiagMPI, CompositeMPI, IdentityMPI,
                      MatKronIdentityMPI, SumMPI, TridiagKronMatMPI, as_matrix)
from mpi_kron_test import linearity_test_MPI, linop_test_MPI
from mpi_vector import KronVectorMPI, DofDistributionMPI
from problem import square, problem_helper

refines = 2


def test_multigrid():
    # Gather the space/time stiffness matrices.
    refines = 2
    heat_eq_mpi = HeatEquationMPI(refines, precond='multigrid')
    M = heat_eq_mpi.M

    for linop in [
            heat_eq_mpi.Kinv_x,
            CompositeLinOp([heat_eq_mpi.Kinv_x, heat_eq_mpi.M_x]),
            CompositeLinOp([heat_eq_mpi.Kinv_x, heat_eq_mpi.A_x])
    ]:
        x = np.random.rand(M)
        mat = as_matrix(linop)

        y_matfree = linop @ x
        y_matvec = mat @ x
        assert np.allclose(y_matfree, y_matvec)


def test_bilforms():
    def test_linop(N, M, linop):
        comm = MPI.COMM_WORLD
        dofs_distr = DofDistributionMPI(MPI.COMM_WORLD, N, M)
        # Create some global vector on root.
        x_mpi = KronVectorMPI(dofs_distr)
        x_glob = None
        if comm.Get_rank() == 0:
            x_glob = np.random.rand(N * M) * 1.0
            y_glob = np.kron(as_matrix(linop.mat_time),
                             as_matrix(linop.mat_space)) @ x_glob
        x_mpi.scatter(x_glob)

        # Apply the vector using MPI
        x_mpi = linop @ x_mpi

        # Check that it is corret.
        x_mpi.gather(x_glob)
        if comm.Get_rank() == 0:
            assert np.allclose(y_glob, x_glob)

    # Gather the space/time stiffness matrices.
    refines = 2
    heat_eq_mpi = HeatEquationMPI(refines)
    N = heat_eq_mpi.N
    M = heat_eq_mpi.M
    for linop in heat_eq_mpi.S.linops:
        test_linop(N, M, linop)


def test_S_apply():
    # Gather the space/time stiffness matrices.
    refines = 2
    heat_eq_mpi = HeatEquationMPI(refines, precond='direct')
    N = heat_eq_mpi.N
    M = heat_eq_mpi.M

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Create some global vector on root.
    dofs_distr = DofDistributionMPI(MPI.COMM_WORLD, N, M)
    x_mpi = KronVectorMPI(dofs_distr)
    x_glob = None
    if rank == 0:
        np.random.seed(0)
        x_glob = np.random.rand(N * M) * 1.0

        # Compare to demo
        _, _, _, S, _, _, _, _, _, _, _ = demo(*square(refines),
                                               precond='direct')
        y_glob = S @ x_glob

        # Compare to np.kron
        S = sum([linop.as_matrix() for linop in heat_eq_mpi.S.linops])
        z_glob = S @ x_glob

    # And apply it using MPI :-)
    x_mpi.scatter(x_glob)
    y_mpi = heat_eq_mpi.S @ x_mpi
    y_mpi.gather(x_glob)
    if rank == 0:
        print(x_glob - z_glob)
        assert (np.allclose(x_glob, z_glob))
        assert (np.allclose(x_glob, y_glob))


def test_solve():
    refines = 2
    for wavelettransform in ['mat', 'interleaved', 'matfree']:
        heat_eq_mpi = HeatEquationMPI(refines, precond='direct')
        N = heat_eq_mpi.N
        M = heat_eq_mpi.M
        dofs_distr = DofDistributionMPI(MPI.COMM_WORLD, N, M)

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        # Solve using demo on root.
        u_glob_mpi = None
        f_glob_mpi = None
        if rank == 0:
            u_glob_mpi = np.empty(N * M)
            f_glob_mpi = np.empty(N * M)

            # Extract f_glob from demo.
            _, _, _, S, _, _, _, _, f_glob_demo, _, _ = demo(*square(refines),
                                                             precond='direct')

            # Solve on root.
            u_glob_demo, _ = PCG(S, scipy.sparse.identity(N * M), f_glob_demo)

        # Solve using mpi.
        def cb(w, residual, k):
            if rank == 0:
                print('.', end='', flush=True)

        u_mpi, _ = PCG(heat_eq_mpi.S,
                       IdentityMPI(dofs_distr),
                       heat_eq_mpi.rhs,
                       callback=cb)

        # Gather the results on root.
        u_mpi.gather(u_glob_mpi)

        # Compare to demo.
        heat_eq_mpi.rhs.gather(f_glob_mpi)
        if rank == 0:
            assert (np.allclose(f_glob_demo, f_glob_mpi))
            assert (np.allclose(u_glob_demo, u_glob_mpi))


def linop_test_apply_MPI(linop_mpi, linop):
    np.random.seed(123123)
    x_mpi = KronVectorMPI(linop_mpi.dofs_distr)

    x_glob = None
    if x_mpi.rank == 0:
        x_glob = np.random.rand(linop_mpi.N * linop_mpi.M)
        y_glob = linop @ x_glob

    x_mpi.scatter(x_glob)
    x_mpi = linop_mpi @ x_mpi
    x_mpi.gather(x_glob)

    if x_mpi.rank == 0:
        assert np.allclose(x_glob, y_glob)


def test_demo():
    for problem in ['square', 'ns']:
        refines = 2
        heat_eq_mpi = HeatEquationMPI(refines,
                                      precond='direct',
                                      problem=problem)

        _, _, WT, S, W, _, P, _, _, _, _ = demo(*problem_helper(
            problem, refines),
                                                precond='direct')
        linop_test_MPI(heat_eq_mpi.WT_S_W, as_matrix(WT @ S @ W))
        linop_test_MPI(heat_eq_mpi.P, as_matrix(P))

        for refines in range(1, 3):
            heat_eq_mpi = HeatEquationMPI(refines,
                                          precond='direct',
                                          problem=problem)
            linearity_test_MPI(heat_eq_mpi.S)
            linearity_test_MPI(heat_eq_mpi.W)
            linearity_test_MPI(heat_eq_mpi.WT)
            linearity_test_MPI(heat_eq_mpi.WT_S_W)
            linearity_test_MPI(heat_eq_mpi.P)

            _, _, WT, S, W, _, P, _, _, _, _ = demo(*problem_helper(
                problem, refines),
                                                    precond='direct')
            linop_test_apply_MPI(heat_eq_mpi.S, S)
            linop_test_apply_MPI(heat_eq_mpi.W, W)
            linop_test_apply_MPI(heat_eq_mpi.WT, WT)
            linop_test_apply_MPI(heat_eq_mpi.WT_S_W, WT @ S @ W)
            linop_test_apply_MPI(heat_eq_mpi.P, P)


def test_preconditioner():
    refines = 3
    heat_eq_mpi = HeatEquationMPI(refines, precond='direct')
    N = heat_eq_mpi.N
    M = heat_eq_mpi.M
    comm = MPI.COMM_WORLD

    # Create random MPI vector.
    dofs_distr = DofDistributionMPI(MPI.COMM_WORLD, N, M)
    w_mpi = KronVectorMPI(dofs_distr)
    w_mpi.X_loc = np.random.rand(w_mpi.X_loc.shape[0], M)

    # Perform Lanczos.
    lanczos_mpi = Lanczos(heat_eq_mpi.WT_S_W, heat_eq_mpi.P, w=w_mpi)

    # Solve without and with preconditioner.
    u_mpi_I, iters_I = PCG(heat_eq_mpi.S, IdentityMPI(dofs_distr),
                           heat_eq_mpi.rhs)
    u_mpi_P, iters_P = PCG(heat_eq_mpi.WT_S_W, heat_eq_mpi.P, heat_eq_mpi.rhs)
    assert iters_P < iters_I

    if w_mpi.rank == 0:
        # Compare to demo
        _, _, WT, S, W, _, P, _, _, _, _ = demo(*square(refines),
                                                precond='direct')
        lanczos_demo = Lanczos(WT @ S @ W, P)
        assert abs(lanczos_mpi.cond() - lanczos_demo.cond()) < 0.4
