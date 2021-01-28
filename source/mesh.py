from ngsolve import Mesh


def construct_interval(N=16, T=1):
    from netgen.meshing import Element0D, Element1D
    from netgen.meshing import Mesh as NGMesh
    from netgen.meshing import MeshPoint, Pnt
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


def construct_2d_square_mesh(nrefines=1):
    from netgen.geom2d import SplineGeometry
    geo = SplineGeometry()
    geo.AddRectangle((0, 0), (1, 1))
    ngmesh = geo.GenerateMesh()
    ngmesh.Refine()
    mesh = Mesh(ngmesh)
    for _ in range(nrefines):
        mesh.Refine()
    return mesh, "default"


def construct_3d_cube_mesh(nrefines=1):
    from netgen.csg import CSGeometry, OrthoBrick, Pnt
    cube = OrthoBrick(Pnt(0, 0, 0), Pnt(1, 1, 1))
    geo = CSGeometry()
    geo.Add(cube, maxh=1)
    ngmesh = geo.GenerateMesh()
    ngmesh.Refine()
    mesh = Mesh(ngmesh)
    for _ in range(nrefines):
        mesh.Refine()
    return mesh, "default"
