import numpy as np
import scipy.sparse
from mpi4py import MPI

from heateq import HeatEquation
from heateq_mpi import HeatEquationMPI
from source.lanczos import Lanczos
from source.linalg import PCG
from source.linop import CompositeLinOp
from source.mpi_kron import IdentityMPI, as_matrix
from source.mpi_kron_test import linearity_test_MPI, linop_test_MPI
from source.mpi_vector import DofDistributionMPI, KronVectorMPI

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


def test_matrices():
    # Gather the space/time stiffness matrices.
    J_time = 4
    J_space = 2
    problem = 'square'
    precond = 'direct'

    heat_eq_mpi = HeatEquationMPI(J_time=J_time,
                                  J_space=J_space,
                                  problem=problem,
                                  wavelettransform='original',
                                  precond=precond)

    # Gather the (dense) S matrix MPI matrix, expensive.
    WT_S_W_mpi = heat_eq_mpi.WT_S_W.as_global_matrix()
    P_mpi = heat_eq_mpi.P.as_global_matrix()

    # Compare on rank 0.
    if MPI.COMM_WORLD.Get_rank() == 0:
        # Compare to demo
        heat_eq = HeatEquation(J_time=J_time,
                               J_space=J_space,
                               problem=problem,
                               precond='direct')
        WT_S_W = as_matrix(heat_eq.WT_S_W)
        P = as_matrix(heat_eq.P)

        assert (np.allclose(WT_S_W, WT_S_W_mpi))
        assert (np.allclose(P, P_mpi))


def test_S_apply():
    # Gather the space/time stiffness matrices.
    J_time = 4
    J_space = 2
    heat_eq_mpi = HeatEquationMPI(J_time=J_time,
                                  J_space=J_space,
                                  precond='direct')
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
        heat_eq = HeatEquation(J_time=J_time,
                               J_space=refines,
                               problem='square',
                               precond='direct')
        y_glob = heat_eq.S @ x_glob

        # Compare to np.kron
        S = sum([linop.as_matrix() for linop in heat_eq_mpi.S.linops])
        z_glob = S @ x_glob

    # And apply it using MPI :-)
    x_mpi.scatter(x_glob)
    y_mpi = heat_eq_mpi.S @ x_mpi
    y_mpi.gather(x_glob)
    if rank == 0:
        assert (np.allclose(x_glob, z_glob))
        assert (np.allclose(x_glob, y_glob))


def test_solve():
    J_time = 4
    J_space = 2
    for precond in ['direct', 'multigrid']:
        heat_eq_mpi = HeatEquationMPI(J_time=J_time,
                                      J_space=J_space,
                                      problem='square',
                                      precond=precond,
                                      smoothsteps=3,
                                      vcycles=4)
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
            heat_eq = HeatEquation(J_time=J_time,
                                   J_space=refines,
                                   problem='square',
                                   precond='direct')

            # Solve on root.
            u_glob_demo, _ = PCG(heat_eq.S, scipy.sparse.identity(N * M),
                                 heat_eq.f)

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
            assert (np.allclose(heat_eq.f, f_glob_mpi))
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
    for problem in ['square', 'cube']:
        refines = 2
        heat_eq_mpi = HeatEquationMPI(refines,
                                      precond='direct',
                                      wavelettransform='original',
                                      problem=problem)

        heat_eq = HeatEquation(problem=problem,
                               J_space=refines,
                               precond='direct')
        linop_test_MPI(heat_eq_mpi.WT_S_W,
                       as_matrix(heat_eq.WT @ heat_eq.S @ heat_eq.W))
        linop_test_MPI(heat_eq_mpi.P, as_matrix(heat_eq.P))

        for refines in range(1, 3):
            heat_eq_mpi = HeatEquationMPI(refines,
                                          precond='direct',
                                          wavelettransform='original',
                                          problem=problem)
            linearity_test_MPI(heat_eq_mpi.S)
            linearity_test_MPI(heat_eq_mpi.W)
            linearity_test_MPI(heat_eq_mpi.WT)
            linearity_test_MPI(heat_eq_mpi.WT_S_W)
            linearity_test_MPI(heat_eq_mpi.P)

            heat_eq = HeatEquation(problem=problem,
                                   J_space=refines,
                                   precond='direct')
            linop_test_apply_MPI(heat_eq_mpi.S, heat_eq.S)
            linop_test_apply_MPI(heat_eq_mpi.W, heat_eq.W)
            linop_test_apply_MPI(heat_eq_mpi.WT, heat_eq.WT)
            linop_test_apply_MPI(heat_eq_mpi.WT_S_W,
                                 heat_eq.WT @ heat_eq.S @ heat_eq.W)
            linop_test_apply_MPI(heat_eq_mpi.P, heat_eq.P)


def test_preconditioner():
    J_time = 4
    J_space = 2
    precond = 'direct'
    heat_eq_mpi = HeatEquationMPI(J_time=J_time,
                                  J_space=J_space,
                                  precond=precond)
    N = heat_eq_mpi.N
    M = heat_eq_mpi.M

    # Create random MPI vector.
    dofs_distr = DofDistributionMPI(MPI.COMM_WORLD, N, M)
    w_mpi = KronVectorMPI(dofs_distr)
    w_mpi.X_loc[:] = np.random.rand(w_mpi.X_loc.shape[0], M)

    # Perform Lanczos.
    lanczos_mpi = Lanczos(heat_eq_mpi.WT_S_W, heat_eq_mpi.P, w=w_mpi)

    # Solve without and with preconditioner.
    u_mpi_I, iters_I = PCG(heat_eq_mpi.S, IdentityMPI(dofs_distr),
                           heat_eq_mpi.rhs)
    u_mpi_P, iters_P = PCG(heat_eq_mpi.WT_S_W, heat_eq_mpi.P, heat_eq_mpi.rhs)
    assert iters_P < iters_I

    if w_mpi.rank == 0:
        # Compare to demo
        heat_eq = HeatEquation(J_time=J_time, J_space=J_space, precond=precond)
        lanczos_demo = Lanczos(heat_eq.WT_S_W, heat_eq.P)
        assert abs(lanczos_mpi.cond() - lanczos_demo.cond()) < 0.1
