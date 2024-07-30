#!/usr/bin/env python3

from collections import namedtuple
import math
import torch as tr

from .raytracer import r_torch, e_torch, a_torch, Operator, find_starts
from .geometry import *

def check(a, b):
    """Helper function for checking equality of two tensors"""
    return tr.allclose(
        tr.asarray(a).type(tr.float32).flatten().squeeze(),
        tr.asarray(b).type(tr.float32).flatten().squeeze(),
        atol=1e-2
    )

def test_r():
    rs = (0.1, 1, 2)

    # ray intersects all shells
    xs = [(-3, 0, 0)]
    rays = [(1, 0, 0)]
    r_t, r_region = r_torch(rs, xs, rays)[:2]
    assert check(r_t, [2.9, 2, 1, 3.1, 4, 5])
    assert check(r_region, [-1, 0, 1, 0, 1, -1])

    # ray goes in opposite direction
    xs = [(-3, 0, 0)]
    rays = [(-1, 0, 0)]
    r_t, r_region = r_torch(rs, xs, rays)[:2]
    assert check(r_t, [-3.1, -4, -5, -2.9, -2, -1])
    assert check(r_region, [-1, 0, 1, 0, 1, -1])

    # ray does not intersect any shells
    xs = [(-3, 0, 0)]
    rays = [(0, 0, 1)]
    r_t, r_region = r_torch(rs, xs, rays)[:2]
    assert tr.all(tr.isinf(r_t))

    # ray tangent to shell
    xs = [(-3, 2, 0), (-3, -2, 0), (-3, -2, 0)]
    rays = [(1, 0, 0), (1, 0, 0), (-1, 0, 0)]
    r_t, r_region = r_torch([2], xs, rays)[:2]
    assert check(r_t, [(3, 3), (3, 3), (-3, -3)])
    assert check(r_region, [(-1, -1), (-1, -1), (-1, -1)])

    # ray through r=0 shell
    xs = [(-3, 0, 0)]
    rays = [(1, 0, 0)]
    r_t, r_region = r_torch([0], xs, rays)[:2]
    assert check(r_t, [3, 3])
    assert check(r_region, [-1, -1])


def test_e():
    e = tr.tensor([tr.pi/6, tr.pi/4])

    # ray intersects all cones once (negative crossing)
    xs = [(-1, 0, 0)]
    rays = [(0, 0, 1)]
    e_t, e_region = e_torch(e, xs, rays)[:2]
    assert check(e_t, [math.sqrt(3), 1, float('inf'), float('inf')])
    assert check(e_region, [-1, 0, -1, 0])

    # ray intersects all cones twice (Z > 0)
    d = 100
    xs = [(-d, 0, 1)]
    rays = [(1, 0, 0)]
    e_t, e_region = e_torch(e, xs, rays)[:2]
    inv3 = 1 / math.sqrt(3)
    assert check(e_t, [d - inv3, d - 1, d + inv3, d + 1])
    assert check(e_region, [-1, 0, 0, -1])

    # ray intersects all cones twice (Z < 0)
    d = 100
    xs = [(-d, 0, -1)]
    rays = [(1, 0, 0)]
    e_t, e_region = e_torch(tr.pi - e, xs, rays)[:2]
    inv3 = 1 / math.sqrt(3)
    assert check(e_t, [d - inv3, d - 1, d + inv3, d + 1])
    assert check(e_region, [0, -1, -1, 0])

    # ray through shadow cones
    xs = [(-1, 0, -1)]
    rays = [(1, 0, 0)]
    e_t, e_region = e_torch(e, xs, rays)[:2]
    assert check(e_t, 4 * [float('inf')])
    assert check(e_region, [0, -1, -1, 0])

    # ray parallel to cone
    xs = [(0, 0, 1)]
    rays = [(1, 0, 1)]
    e_t, e_region = e_torch([tr.pi / 4], xs, rays)[:2]
    assert check(e_t, [-1 / math.sqrt(2), float('inf')])
    assert check(e_region, [-1, -1])

    # ray on cone
    xs = [(-1, 0, 1)]
    rays = [(1, 0, -1)]
    e_t, e_region = e_torch([tr.pi / 4], xs, rays)[:2]
    # assert check(e_t, [float('inf'), float('inf')])
    assert check(e_t, [-float('inf'), -float('inf')])
    assert check(e_region, [-1, -1])

    # ray tangent to cone
    xs = [(1, 1, 1)]
    rays = [(0, -1, 0)]
    e_t, e_region = e_torch([tr.pi / 4], xs, rays)[:2]
    assert check(e_t, [1, 1])
    assert check(e_region, [-2, -2])

    # ray through origin
    xs = [(-1, 0, 0)]
    rays = [(1, 0, 0)]
    e_t, e_region = e_torch(e, xs, rays)[:2]
    # assert check(e_t, [1, 1, float('inf'), float('inf')])
    assert check(e_t, [1, 1, 1, 1])
    # FIXME
    # assert check(e_region, [0, -1, -1, 0])

    # ray through e=0, e=π cones
    xs = [(-1, 0, 0)]
    rays = [(1, 0, 0)]
    e_t, e_region = e_torch([0, tr.pi], xs, rays)[:2]
    # FIXME:
    # assert check(e_t, [1, 1, float('inf'), float('inf')])
    # assert check(e_region, [-1, 0, -1, 0])


def test_a():
    a_b = [tr.pi/4, tr.pi/2]

    # ray intersects all planes once (negative crossing)
    xs = [(-1, 1, 0)]
    rays = [(1, 0, 0)]
    a_t, a_region = a_torch(a_b, xs, rays)[:2]
    assert check(a_t, [2, 1])
    assert check(a_region, [-1, 0])

    # ray intersects all planes once (positive crossing)
    xs = [(-1, 1, 0)]
    rays = [(-1, 0, 0)]
    a_t, a_region = a_torch(a_b, xs, rays)[:2]
    assert check(a_t, [-2, -1])
    assert check(a_region, [0, -1])

    # ray intersects no planes
    xs = [(-1, -1, 0)]
    rays = [(1, 0, 0)]
    a_t, a_region = a_torch(a_b, xs, rays)[:2]
    assert check(a_t, [float('inf'), float('inf')])

    # ray parallel to plane
    xs = [(0, 1, 0)]
    rays = [(1, 0, 0)]
    a_t, a_region = a_torch([0], xs, rays)[:2]
    assert check(a_t.abs(), [float('inf')])

    # ray through origin
    xs = [(-1, 0, 0)]
    rays = [(1, 0, 0)]
    a_t, a_region = a_torch(a_b, xs, rays)[:2]
    assert check(a_t, [1, 1])
    assert check(a_region, [-1, 0])

    # ray on plane
    xs = [(0, 1, 0)]
    rays = [(0, -1, 0)]
    a_t, a_region = a_torch([tr.pi/2], xs, rays)[:2]
    # FIXME:
    # assert check(a_t, [1, 1])
    # assert check(a_region, [1, 0])


def test_sphericalgrid():
    grid = SphericalGrid(shape=(10, 11, 12))
    assert (len(grid.rs_b), len(grid.e_b), len(grid.a_b)) == (11, 12, 13)
    grid = SphericalGrid(rs_b=[1, 2], e_b=[1, 2, 3], a_b=[1, 2, 3, 4])
    assert grid.shape == (1, 2, 3)
    # check grid boundaries and centers
    def check_bounds(grid):
        assert len(grid.rs) == len(grid.rs_b) - 1
        assert len(grid.e) == len(grid.e_b) - 1
        assert len(grid.a) == len(grid.a_b) - 1
        assert all(grid.rs > grid.rs_b[:-1])
        assert all(grid.e > grid.e_b[:-1])
        assert all(grid.a > grid.a_b[:-1])
        assert all(grid.rs < grid.rs_b[1:])
        assert all(grid.e < grid.e_b[1:])
        assert all(grid.a < grid.a_b[1:])

    check_bounds(grid)
    check_bounds(
        SphericalGrid(
            shape=(10, 11, 12),
            size_r=(1, 10), size_e=(0, tr.pi), size_a=(0, 2*tr.pi),
            spacing='log',

        )
    )

def test_find_starts():
    grid = SphericalGrid(shape=(5, 5, 1))
    s = find_starts(grid, [0, 0, 100])
    assert check(s, [-1, 0, 0])
    s = find_starts(grid, [0, 0, -100])
    assert check(s, [-1, 4, 0])

    grid = SphericalGrid(shape=(5, 5, 5))
    s = find_starts(grid, [100, 0, 0])
    assert check(s, [-1, 2, 2])


def test_operator():
    # trace through center of solid sphere
    grids = [
        SphericalGrid(shape=(50, 50, 50), size_r=(3, 25), size_e=(0, tr.pi), size_a=(-tr.pi, tr.pi)),
        SphericalGrid(shape=(4, 4, 4)),
        SphericalGrid(shape=(1, 4, 4)),
        SphericalGrid(shape=(4, 1, 4)),
        SphericalGrid(shape=(4, 4, 1)),
    ]
    u = 0.001
    xs = [
        [-100, u, u],
        [u, -100, u],
        [u, u, -100],
        [-100, 0, u],
        [0, -100, u],
        [0, u, -100],
        [-100, u, 0],
        [u, -100, 0],
        [u, 0, -100],
        [5, 0, 0],
    ]
    rays = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        # ray just barely glances cone
        [-0.99998629093170166016,  0.00413372274488210678, 0.00321511807851493359],
    ]
    for grid in grids:
        geom = ViewGeom(xs, rays)
        op = Operator(grid, geom, _flatten=False)
        d = tr.ones(grid.shape)
        result = op(d)
        diam = 2 * (grid.size[0][1] - grid.size[0][0])
        ray_success = tr.isclose(result, tr.tensor(diam, dtype=result.dtype))
        fail_str = f"Failure for grid={grid} for ray #s {tr.where(ray_success == False)[0].tolist()}"
        assert all(tr.isclose(result, tr.tensor(diam, dtype=result.dtype), atol=1e-2)), fail_str

    # trace ray through center of hollow sphere
    geom = ViewGeom([-100, 0, 0], [1, 0, 0])
    grid = SphericalGrid(shape=(25, 25, 25), size_r=(5, 10))
    op = Operator(grid, geom, _flatten=False)

def test_conerectgeom():
    g = ConeRectGeom((11, 11), (4, 0, 1), fov=(23, 45))

    # check fov angles
    assert check(tr.dot(g.rays[5, 0], g.rays[5, -1]), tr.cos(tr.deg2rad(g.fov[1])))
    assert check(tr.dot(g.rays[0, 5], g.rays[-1, 5]), tr.cos(tr.deg2rad(g.fov[0])))
    # check lookdir
    assert check(g.rays[5, 5], g.lookdir)

    # single pixel detector
    g = ConeRectGeom((1, 1), (1, 0, 0), (-1, 0, 0), (0, 1, 0), fov=(23, 45))
    # check lookdir
    assert check(g.rays[0, 0], g.lookdir)
    # generate wireframe
    g._wireframe


def test_conecircgeom():
    g = ConeCircGeom((11, 11), (1, 0, 0), (-1, 0, 0), (0, 1, 0), fov=45)

    # check fov angles
    assert check(tr.dot(g.rays[-1, 0], g.rays[-1, 5]), tr.cos(tr.deg2rad(g.fov)))
    # check look dir
    assert check(g.rays[0, 0], g.lookdir)

    # single pixel detector
    g = ConeCircGeom((1, 1), (1, 0, 0), (-1, 0, 0), (0, 1, 0), fov=45)
    # check lookdir
    assert check(g.rays[0, 0], g.lookdir)
    # generate wireframe
    g._wireframe

def test_parallelgeom():
    g = ParallelGeom((11, 11), (4, 0, 1), size=(2, 3))

    # check ray separation
    assert check(
        tr.linalg.norm(g.ray_starts[5, 0] - g.ray_starts[5, -1]),
        g.size[1]
    )
    assert check(
        tr.linalg.norm(g.ray_starts[0, 5] - g.ray_starts[-1, 5]),
        g.size[0]
    )
    # check lookdir
    assert all((g.rays == g.lookdir).flatten())

    # single pixel detector
    g = ParallelGeom((1, 1), (1, 0, 0), (-1, 0, 0), (0, 1, 0))
    # check lookdir
    assert check(g.rays[0, 0], g.lookdir)
    # generate wireframe
    g._wireframe


def test_viewgeom():
    # not much to test here.  just instantiate a ViewGeom with random LOS's
    rays = tr.rand((4, 4, 3))
    ray_starts=tr.tensor((10, 0, 0)).broadcast_to(rays.shape)
    g = ViewGeom(
        rays=rays,
        ray_starts=ray_starts
    )
    # generate wireframe
    g._wireframe