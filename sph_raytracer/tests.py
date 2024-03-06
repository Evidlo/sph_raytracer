#!/usr/bin/env python3

from collections import namedtuple
import math
import torch as tr

from .raytracer import r_torch, e_torch, a_torch, trace_indices, find_starts
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
    assert tr.all(tr.isnan(r_t))

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
    phis = tr.tensor([tr.pi/6, tr.pi/4])

    # ray intersects all cones once (negative crossing)
    xs = [(-1, 0, 0)]
    rays = [(0, 0, 1)]
    e_t, e_region = e_torch(phis, xs, rays)[:2]
    assert check(e_t, [math.sqrt(3), 1, float('inf'), float('inf')])
    assert check(e_region, [-1, 0, -1, 0])

    # ray intersects all cones twice (Z > 0)
    d = 100
    xs = [(-d, 0, 1)]
    rays = [(1, 0, 0)]
    e_t, e_region = e_torch(phis, xs, rays)[:2]
    inv3 = 1 / math.sqrt(3)
    assert check(e_t, [d - inv3, d - 1, d + inv3, d + 1])
    assert check(e_region, [-1, 0, 0, -1])

    # ray intersects all cones twice (Z < 0)
    d = 100
    xs = [(-d, 0, -1)]
    rays = [(1, 0, 0)]
    e_t, e_region = e_torch(tr.pi - phis, xs, rays)[:2]
    inv3 = 1 / math.sqrt(3)
    assert check(e_t, [d - inv3, d - 1, d + inv3, d + 1])
    assert check(e_region, [0, -1, -1, 0])

    # ray through shadow cones
    xs = [(-1, 0, -1)]
    rays = [(1, 0, 0)]
    e_t, e_region = e_torch(phis, xs, rays)[:2]
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
    e_t, e_region = e_torch(phis, xs, rays)[:2]
    # assert check(e_t, [1, 1, float('inf'), float('inf')])
    assert check(e_t, [1, 1, 1, 1])
    # FIXME
    # assert check(e_region, [0, -1, -1, 0])

    # ray through phi=0, phi=π cones
    xs = [(-1, 0, 0)]
    rays = [(1, 0, 0)]
    e_t, e_region = e_torch([0, tr.pi], xs, rays)[:2]
    # FIXME:
    # assert check(e_t, [1, 1, float('inf'), float('inf')])
    # assert check(e_region, [-1, 0, -1, 0])


def test_a():
    thetas = [tr.pi/4, tr.pi/2]

    # ray intersects all planes once (negative crossing)
    xs = [(-1, 1, 0)]
    rays = [(1, 0, 0)]
    a_t, a_region = a_torch(thetas, xs, rays)[:2]
    assert check(a_t, [2, 1])
    assert check(a_region, [-1, 0])

    # ray intersects all planes once (positive crossing)
    xs = [(-1, 1, 0)]
    rays = [(-1, 0, 0)]
    a_t, a_region = a_torch(thetas, xs, rays)[:2]
    assert check(a_t, [-2, -1])
    assert check(a_region, [0, -1])

    # ray intersects no planes
    xs = [(-1, -1, 0)]
    rays = [(1, 0, 0)]
    a_t, a_region = a_torch(thetas, xs, rays)[:2]
    assert check(a_t, [float('inf'), float('inf')])

    # ray parallel to plane
    xs = [(0, 1, 0)]
    rays = [(1, 0, 0)]
    a_t, a_region = a_torch([0], xs, rays)[:2]
    assert check(a_t.abs(), [float('inf')])

    # ray through origin
    xs = [(-1, 0, 0)]
    rays = [(1, 0, 0)]
    a_t, a_region = a_torch(thetas, xs, rays)[:2]
    assert check(a_t, [1, 1])
    assert check(a_region, [-1, 0])

    # ray on plane
    xs = [(0, 1, 0)]
    rays = [(0, -1, 0)]
    a_t, a_region = a_torch([tr.pi/2], xs, rays)[:2]
    # FIXME:
    # assert check(a_t, [1, 1])
    # assert check(a_region, [1, 0])


def test_spherical_vol():
    vol = SphericalVol(shape=(10, 11, 12))
    assert (len(vol.rs), len(vol.phis), len(vol.thetas)) == (11, 12, 13)
    vol = SphericalVol(rs=[1, 2], phis=[1, 2, 3], thetas=[1, 2, 3, 4])
    assert vol.shape == (1, 2, 3)

def test_find_starts():
    vol = SphericalVol(shape=(5, 5, 1))
    s = find_starts(vol, [0, 0, 100])
    assert check(s, [-1, 0, 0])
    s = find_starts(vol, [0, 0, -100])
    assert check(s, [-1, 4, 0])

    vol = SphericalVol(shape=(5, 5, 5))
    s = find_starts(vol, [100, 0, 0])
    assert check(s, [-1, 2, 2])


def test_trace_indices():
    # trace through center of solid sphere
    vols = [
        SphericalVol(shape=(50, 50, 50), size=((3, 25), (0, tr.pi), (-tr.pi, tr.pi))),
        SphericalVol(shape=(4, 4, 4)),
        SphericalVol(shape=(1, 4, 4)),
        SphericalVol(shape=(4, 1, 4)),
        SphericalVol(shape=(4, 4, 1)),
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
    for vol in vols:
        regions, lens = trace_indices(vol, xs, rays)
        d = tr.ones(vol.shape)
        result = (d[regions] * lens).sum(axis=-1)
        diam = 2 * (vol.size[0][1] - vol.size[0][0])
        ray_success = tr.isclose(result, tr.tensor(diam, dtype=result.dtype))
        fail_str = f"Failure for vol={vol} for ray #s {tr.where(ray_success == False)[0].tolist()}"
        assert all(tr.isclose(result, tr.tensor(diam, dtype=result.dtype), atol=1e-2)), fail_str

def test_conerectgeom():
    g = ConeRectGeom((11, 11), (1, 0, 0), (-1, 0, 0), (0, 1, 0), fov=(23, 45))

    # check fov angles
    assert check(tr.dot(g.rays[5, 0], g.rays[5, -1]), tr.cos(tr.deg2rad(g.fov[0])))
    assert check(tr.dot(g.rays[0, 5], g.rays[-1, 5]), tr.cos(tr.deg2rad(g.fov[1])))
    # check lookdir
    assert check(g.rays[5, 5], g.lookdir)

    # single pixel detector
    g = ConeRectGeom((1, 1), (1, 0, 0), (-1, 0, 0), (0, 1, 0), fov=(23, 45))
    # check lookdir
    assert check(g.rays[0, 0], g.lookdir)


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