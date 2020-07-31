from ngsolve import Mesh, Draw, mpi_world


def construct_3d_cube_mesh(nrefines=1):
    from netgen.csg import OrthoBrick, Pnt, CSGeometry
    cube = OrthoBrick(Pnt(0, 0, 0), Pnt(1, 1, 1))
    geo = CSGeometry()
    geo.Add(cube, maxh=1.0)
    ngmesh = geo.GenerateMesh()
    for _ in range(nrefines):
        ngmesh.Refine()
    mesh = Mesh(ngmesh)
    return mesh, "default"


def construct_2d_square_mesh(nrefines=1):
    from netgen.geom2d import SplineGeometry
    geo = SplineGeometry()
    geo.AddRectangle((0, 0), (1, 1))
    ngmesh = geo.GenerateMesh()
    for _ in range(nrefines):
        ngmesh.Refine()
    mesh = Mesh(ngmesh)
    return mesh, "default"


def construct_3d_shaft_mesh(nrefines=0):
    from netgen.meshing import Mesh as NGMesh, meshsize
    ngmesh = NGMesh(dim=3)
    ngmesh.Load('shaft.vol')
    ngmesh.GenerateVolumeMesh()
    for _ in range(nrefines):
        ngmesh.Refine()
    mesh = Mesh(ngmesh)
    Draw(mesh)
    return mesh, "default"


def construct_2d_navierstokes_mesh(maxh=0.07):
    from netgen.geom2d import SplineGeometry
    geo = SplineGeometry()
    geo.AddRectangle((0, 0), (2, 0.41),
                     bcs=("wall", "outlet", "wall", "inlet"))
    geo.AddCircle((0.2, 0.2),
                  r=0.05,
                  leftdomain=0,
                  rightdomain=1,
                  bc="cyl",
                  maxh=0.02)
    return Mesh(geo.GenerateMesh(maxh=maxh)), "wall|cyl|inlet"


def construct_interval(N=16, T=1):
    from netgen.meshing import Mesh as NGMesh, Element0D, Element1D, MeshPoint, Pnt
    ngmesh = NGMesh(dim=1)
    pids = []
    for i in range(N + 1):
        pids.append(ngmesh.Add(MeshPoint(Pnt(T * i / N, 0, 0))))
    for i in range(N):
        ngmesh.Add(Element1D([pids[i], pids[i + 1]], index=1))
    ngmesh.Add(Element0D(pids[0], index=1))
    ngmesh.Add(Element0D(pids[N], index=2))
    ngmesh.SetBCName(0, "start")
    ngmesh.SetBCName(1, "end")
    return Mesh(ngmesh)
