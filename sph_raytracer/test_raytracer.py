#!/usr/bin/env python3

import torch as tr

from .raytracer import Operator
from .geometry import *

def test_operator_static():
    # trace through center of solid sphere
    grids = [
        SphericalGrid(shape=(50, 50, 50), size_r=(3, 25), size_e=(0, tr.pi), size_a=(-tr.pi, tr.pi)),
        SphericalGrid(shape=(4, 4, 4)),
        SphericalGrid(shape=(1, 4, 4)),
        SphericalGrid(shape=(4, 1, 4)),
        SphericalGrid(shape=(4, 4, 1)),
    ]
    # offset traced ray a small amount in several directions to check for rounding errors
    u = 0.001
    ray_starts = [
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
        # ray just barely glances elevation cone
        [-0.99998629093170166016,  0.00413372274488210678, 0.00321511807851493359],
    ]
    for grid in grids:
        geom = ViewGeom(ray_starts, rays)
        op = Operator(grid, geom)
        d = tr.ones(grid.shape)
        result = op(d)
        diam = 2 * (grid.size[0][1] - grid.size[0][0])
        ray_success = tr.isclose(result, tr.tensor(diam, dtype=result.dtype))
        fail_str = f"Failure for grid={grid} for ray #s {tr.where(ray_success == False)[0].tolist()}"
        assert all(tr.isclose(result, tr.tensor(diam, dtype=result.dtype), atol=1e-2)), fail_str

    # trace ray through center of hollow sphere
    geom = ViewGeom([-100, 0, 0], [1, 0, 0])
    grid = SphericalGrid(shape=(25, 25, 25), size_r=(5, 10))
    op = Operator(grid, geom)
    # trace multidimensional static volume
    result = op(tr.rand((5,) + grid.shape))
    assert result.shape == (5,), "Incorrect shape for multi-channel volume"


# check operator result shapes under various conditions
def test_operator_shape():
    # trace through center of solid sphere
    grids = [
        # static grid and static input
        [SphericalGrid((2, 3, 4)),     tr.rand((2, 3, 4))],
        # static grid and static multichannel input
        [SphericalGrid((2, 3, 4)),     tr.rand((10, 2, 3, 4))],
        # dynamic grid and dynamic input
        [SphericalGrid((10, 2, 3, 4)), tr.rand((10, 2, 3, 4))],
    ]

    geom = ConeRectGeom(shape:=(64, 64), (1, 0, 0))
    for grid, d in grids:
        op = Operator(grid, geom)
        result = op(d)
        # shape of channel/time dimensions (i.e. nonspatial dimensions)
        chan_time = d.shape[:-3]
        fail_str = f"Failure for grid={grid} and input={d.shape}"
        assert result.shape == chan_time + shape, f"Invalid shape: {fail_str}"

# raytracer regression bugs
def test_buggy_los():
    # regression tests for specific LOS/grids that have given incorrect results
    grids = [
        SphericalGrid(shape=(1, 2, 1), size_r=(0, 25)),
    ]
    # densities
    ds = [
        tr.tensor([[[1.0], [0]]]) # upper hemisphere filled
    ]
    u = 0.001
    ray_starts = [
        [-200, u, u],
    ]
    rays = [
        [1, 0, 0],
    ]
    correct_results = [
        50
    ]
    test_cases = zip(grids, ds, ray_starts, rays, correct_results)
    for grid, d, ray_start, ray, correct_result in test_cases:
        geom = ViewGeom(ray_start, ray)
        op = Operator(grid, geom)
        result = op(d)
        ray_success = tr.isclose(result, tr.tensor(correct_result, dtype=result.dtype))
        fail_str = f"Failure for grid={grid} for ray #s {tr.where(ray_success == False)[0].tolist()}"
        assert tr.isclose(result, tr.tensor(correct_result, dtype=result.dtype), atol=1e-2), fail_str

    # trace ray through center of hollow sphere
    geom = ViewGeom([-100, 0, 0], [1, 0, 0])
    grid = SphericalGrid(shape=(25, 25, 25), size_r=(5, 10))
    op = Operator(grid, geom)