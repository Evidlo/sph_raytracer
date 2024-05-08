#!/usr/bin/env python3

from .plotting import image_stack, preview3d, color_negative
from .geometry import ConeCircGeom, ConeRectGeom, SphericalGrid, ParallelGeom
import torch as tr
import matplotlib

def test_preview3d():
    vol = tr.rand((50, 50, 50))
    grid = SphericalGrid(shape=vol.shape)
    result = preview3d(vol, grid)
    assert result.shape == (20, 256, 256), "Incorrect preview3d shape"
    result = preview3d(color_negative(vol), grid)
    assert result.shape == (20, 256, 256, 3), "Incorrect preview3d shape"

def test_image_stack():
    # test plotting stacks of images with different view geometries
    vg = ConeCircGeom((10, 10), (1, 0, 0))
    vgc = vg + vg
    images = tr.rand(vgc.shape)
    anim = image_stack(images, vgc)
    anim = image_stack(images, vg)
    vg = ConeRectGeom(vg.shape, (1, 0, 0))
    anim = image_stack(images, vg)

def test_viewgeom_plot():
    geoms = [
        ConeCircGeom((11, 11), (1, 0, 0)),
        ConeRectGeom((11, 11), (1, 0, 0)),
        ParallelGeom((11, 11), (1, 0, 0)),
    ]
    matplotlib.use('Agg')
    for g in geoms:
        g.plot()