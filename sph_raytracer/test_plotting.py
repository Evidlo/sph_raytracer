#!/usr/bin/env python3

from .plotting import image_stack, preview3d
import torch as t
import matplotlib

def test_preview3d():
    vol = t.rand((50, 50, 50))
    result = preview3d(vol)
    assert result.shape == (20, 256, 256), "Incorrect preview3d shape"