from ngsolve import mpi_world
from mesh import *
import numpy as np


def neumuller_smears(nrefines, N_time=None):
    #u(t,x,y,z) =e^(−3π2t)sin(πx) sin(πy) sin(πz).
    mesh_space, bc = construct_3d_cube_mesh(nrefines=nrefines)
    if not N_time:
        N_time = 2**int(
            np.log(mesh_space.nv**(1.0 / mesh_space.dim)) / np.log(2) + 0.5)
    # EIGENLIJK T=0.1 maar dat verneukt de solver
    mesh_time = construct_interval(N=N_time)

    from ngsolve import sin, x, y, z
    data = {'g': [], 'u0': sin(np.pi * x) * sin(np.pi * y) * sin(np.pi * z)}
    return mesh_space, bc, mesh_time, data, "neumuller_smears"


def cube(nrefines, N_time=None):
    mesh_space, bc = construct_3d_cube_mesh(nrefines=nrefines)
    if not N_time:
        N_time = 2**int(
            np.log(mesh_space.nv**(1.0 / mesh_space.dim)) / np.log(2) + 0.5)
    mesh_time = construct_interval(N=N_time)
    from ngsolve import x, y, z
    t = x
    data = {
        'g': [(2 * t, x * (1 - x) * y * (1 - y) * z * (1 - z)),
              (2 * (t * t + 1), y * (1 - y) * z * (1 - z) + x * (1 - x) * y *
               (1 - y) + x * (1 - x) * z * (1 - z))],
        'u0':
        x * (1 - x) * y * (1 - y) * z * (1 - z)
    }
    return mesh_space, bc, mesh_time, data, "cube"


def shaft(nrefines, N_time=None):
    mesh_space, bc = construct_3d_shaft_mesh(nrefines=nrefines)
    if not N_time:
        N_time = 2**int(
            np.log(mesh_space.nv**(1.0 / mesh_space.dim)) / np.log(2) + 0.5)
    mesh_time = construct_interval(N=N_time)
    from ngsolve import x
    data = {'g': [], 'u0': x}
    return mesh_space, bc, mesh_time, data, "shaft"


def square(nrefines, N_time=None):
    mesh_space, bc = construct_2d_square_mesh(nrefines=nrefines)
    if not N_time:
        N_time = 2**int(
            np.log(mesh_space.nv**(1.0 / mesh_space.dim)) / np.log(2) + 0.5)
    mesh_time = construct_interval(N=N_time)

    from ngsolve import sin, x, y
    data = {'g': [], 'u0': sin(np.pi * x) * sin(np.pi * y)}
    return mesh_space, bc, mesh_time, data, "square"
