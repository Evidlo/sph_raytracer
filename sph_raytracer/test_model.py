#!/usr/bin/env python3

from sph_raytracer.model import *
from sph_raytracer import SphericalGrid
import torch as t

def test_instantiate_models():
    """Check that models can be instantiated and generate a volume"""
    g = SphericalGrid()

    for model in (FullyDenseModel, CubesModel, AxisAlignmentModel):
        m = model(g)
        c = t.rand(m.coeffs_shape)
        result = m(c)
        assert result.shape == g.shape, f"Invalid volume shape returned by model {m}"