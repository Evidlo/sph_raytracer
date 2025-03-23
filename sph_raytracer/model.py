#!/usr/bin/env python3

import torch as t

from sph_raytracer.geometry import SphericalGrid

class Model:
    """A parameterized model for a volume.  Subclass this class and provide
    implementations for the unimplemented functions to make custom models

    Properties:
        coeffs_shape (tuple): Shape of input coeffs

    Usage:
        g = SphericalGrid(...)
        m = Model(g)

        vol = m(coeffs)
    """

    def __init__(self, grid: SphericalGrid):
        """Do any model setup here"""
        raise NotImplementedError

    def __call__(self, coeffs):
        """Generate volume density from parameters.

        Args:
            coeffs (ndarray or tensor): array of shape self.shape
        """
        raise NotImplementedError

    @property
    def coeffs_shape(self):
        """Shape of coeffs"""
        raise NotImplementedError

    def __repr__(self):
        return f'{self.__class__.__name__}({tuple(self.grid.shape)})'


class FullyDenseModel(Model):
    """Parameters themselves are volume density values"""
    def __init__(self, grid: SphericalGrid):
        self.grid = grid

    def __call__(self, coeffs):
        return coeffs

    @property
    def coeffs_shape(self):
        return self.grid.shape


class CubesModel(Model):
    """Test model with two boxes in spherical coordinates"""
    def __init__(self, grid: SphericalGrid):
        self.grid = grid
        self.volume = t.zeros(grid.shape)
        r0, r1 = grid.shape.r * t.tensor((.333, .666))
        e00, e01 = grid.shape.e * t.tensor((.2, .3))
        e10, e11 = grid.shape.e * t.tensor((.7, .9))
        a0, a1 = grid.shape.a * t.tensor((.4, .6))

        r0, r1 = int(r0), int(r1)
        e00, e01, e10, e11 = int(e00), int(e01), int(e10), int(e11)
        a0, a1 = int(a0), int(a1)

        # self.volume[int(r0):int(r1), int(e00):int(e01), int(a0):int(a1)] = 1
        # self.volume[int(r0):int(r1), int(e10):int(e11), int(a0):int(a1)] = 1
        self.volume[r0:r1, e00:e01, a0:a1] = 1
        self.volume[r0:r1, e10:e11, a0:a1] = 1

        self.r0, self.r1 = r0, r1
        self.e00, self.e01, self.e10, self.e11 = e00, e01, e10, e11
        self.a0, self.a1 = a0, a1

    def __call__(self, coeffs):
        return self.volume

    @property
    def coeffs_shape(self):
        return ()


class AxisAlignmentModel(Model):
    """Test Model to verify that tomographic projections are not mirrored
    Z
    |
    |
    |   Y
    |  /
    | /
    |/
    .--X

    """
    def __init__(self, grid: SphericalGrid):
        self.grid = grid
        self.volume = t.zeros(grid.shape)

        # X axis
        self.volume[:grid.shape.r//3, grid.shape.e//2, 0] = 1
        # Y axis
        self.volume[:grid.shape.r//2, grid.shape.e//2, (grid.shape.a*3)//4] = 1
        # Z axis
        self.volume[:, 0, :] = 1

    def __call__(self, coeffs):
        return self.volume

    @property
    def coeffs_shape(self):
        return ()