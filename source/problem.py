import numpy as np

from .mesh import (construct_2d_square_mesh, construct_3d_cube_mesh,
                   construct_interval)


def square(J_space, J_time=None):
    # Solution u(t,x,y) = exp(−2 π^2 t) sin(πx) sin(πy) on [0,1]^2.
    mesh_space, bc = construct_2d_square_mesh(nrefines=J_space)
    if not J_time:
        J_time = J_space

    N_time = 2**int(J_time + 0.5)
    mesh_time = construct_interval(N=N_time)

    from ngsolve import sin, x, y
    data = {'g': [], 'u0': sin(np.pi * x) * sin(np.pi * y)}
    return mesh_space, bc, mesh_time, data, "square"


def cube(J_space, J_time=None):
    # Solution u(t,x,y) = exp(−3 π^2 t) sin(πx) sin(πy) \sin(πz) on [0,1]^3.
    mesh_space, bc = construct_3d_cube_mesh(nrefines=J_space)
    if not J_time:
        J_time = J_space

    N_time = 2**int(J_time + 0.5)
    mesh_time = construct_interval(N=N_time)

    from ngsolve import sin, x, y, z
    data = {'g': [], 'u0': sin(np.pi * x) * sin(np.pi * y) * sin(np.pi * z)}
    return mesh_space, bc, mesh_time, data, "cube"


def problem_helper(problem, J_space, J_time=None):
    if problem == 'square':
        return square(J_space, J_time)
    elif problem == 'cube':
        return cube(J_space, J_time)
    else:
        assert (False)
