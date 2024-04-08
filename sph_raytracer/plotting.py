#!/usr/bin/env python3

import copy
from matplotlib import animation
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter
import numpy as np
import torch as tr
from itertools import chain, repeat

from .geometry import SphericalGrid, ConeRectGeom, ConeCircGeom, ViewGeomCollection
from .raytracer import Operator

# def add_anim(self, other):
#     """Merge two Animations by chaining calls to _func and _init_func

#     https://github.com/matplotlib/matplotlib/issues/27810
#     """

#     # if self._interval != other._interval:
#     #     # FIXME: possible to check if number of frames equal?
#     #     raise ValueError("Animations must have equal interval")

#     # chain _step methods
#     orig_step = self._step
#     def chained_step(*args):
#         # stop iterating if either animation hits end
#         return orig_step(*args) and other._step(*args)
#     self._step = chained_step

#     # pause the other animation
#     other.pause()

#     return self

# class CollectedAnimation(animation.TimedAnimation):
#     """
#     `TimedAnimation` subclass for merging animations.
#     """
#     def __init__(self, fig, animations, *args, **kwargs):
#         self.animations = animations

#         super().__init__(fig, *args, **kwargs)

#         # pause the animations
#         for a in  animations:
#             a.pause()

#     def _step(self, *args):
#         # stop iterating if any animation hits end
#         return all(a._step(*args) for a in self.animations)

#     def new_frame_seq(self):
#         return chain(a.new_frame_seq() for a in self.animations)

#     def new_saved_frame_seq(self):
#         return chain(a.new_saved_frame_seq() for a in self.animations)

#     def _draw_next_frame(self, *args, **kwargs):
#         for a in self.animations:
#             a._draw_next_frame(*args, **kwargs)

#     def _init_draw(self):
#         for a in self.animations:
#             a._init_draw()


# class SummableAnimation(animation.FuncAnimation):

#     __add__ = add_anim

#     def __radd__(self, other):
#         return self.__add__(other)


# class SummableArtistAnimation(animation.ArtistAnimation):

#     __add__ = add_anim

#     def __radd__(self, other):
#         return self.__add__(other)


def image_stack(images, geom=None, ax=None, colorbar=False, **kwargs):
    """Animate a stack of images

    Args:
        images (ndarray or tensor): array of shape (num_images, width, height)
            for an animated sequence or (width, height) for a single image
        geom (ConeRectGeom, ConeCircGeom or ViewGeomCollection): view geometry
            for labelling FOV.  If ViewGeomCollection, all geometries should be
            of the same type
        ax (matplotlib Axes, optional): existing Axes object to use
        colorbar (bool): include a colorbar
        **kwargs: arguments to pass to plot

    Returns:
        matplotlib.animation.ArtistAnimation
    """
    polar = (
        isinstance(geom, ConeCircGeom) or
        (isinstance(geom, ViewGeomCollection) and isinstance(geom.geoms[0], ConeCircGeom))
    )
    if ax is None:
        fig = plt.figure(figsize=(3, 3))
        ax = fig.add_subplot(polar=polar)

    # detach from GPU if necessary
    if isinstance(images, tr.Tensor):
        images = images.detach().cpu().numpy()

    deg_format = EngFormatter(unit=u"°", sep="")

    if polar:
        def imshow(img, geom, **kwargs):
            # r_lin = np.logspace(np.log10(3), np.log10(25), 100)
            r_lin = np.linspace(0, geom.fov/2, geom.shape[0])
            theta_lin = np.linspace(0, 2*np.pi, geom.shape[1])
            theta, r = np.meshgrid(theta_lin, r_lin)
            ax.yaxis.set_major_formatter(deg_format)
            return ax.pcolormesh(theta, r, img, **kwargs)
    else:
        def imshow(img, geom, **kwargs):
            extent = (-geom.fov[1]/2, geom.fov[1]/2, -geom.fov[0]/2, geom.fov[0]/2) if geom is not None else None
            ax.xaxis.set_major_formatter(deg_format)
            ax.yaxis.set_major_formatter(deg_format)
            return ax.imshow(img, extent=extent, **kwargs)

    vmin, vmax = images.min(), images.max()
    if images.ndim == 3:
        # use same ViewGeom for all images if a ViewGeomCollection not provided
        geom = geom if isinstance(geom, ViewGeomCollection) else repeat(geom)
        artists = [[imshow(im, g, animated=True, vmin=vmin, vmax=vmax, **kwargs)] for im, g in zip(images, geom)]
        result = animation.ArtistAnimation(ax.figure, artists, interval=200)
    elif images.ndim == 2:
        artists = [imshow(images, geom, **kwargs)]
        result = artists[0]
    else:
        raise ValueError("Invalid images shape")

    if colorbar:
        ax_col = ax.twinx()
        ax_col.tick_params(which="both", right=False, labelright=False)
        plt.colorbar(artists[0][0], ax=ax_col)

    return result


def color_negative(x):
    """Convert grayscale multidimensional tensor with negative values to RGB version

    Args:
        x (tensor): multidimensional grayscale array of shape (A, B, ..., Z)

    Returns:
        colored (tensor): multidimensional array of shape (A, B, ..., Z, 3)
    """
    neg = tr.clone(x)
    pos = tr.clone(x)
    pos[pos < 0] = 0
    neg[neg >= 0] = 0
    neg *= -1

    return tr.stack((pos, neg, tr.zeros(pos.shape, device=x.device)), axis=-1)


def preview3d(volume, grid, positions=20, shape=(256, 256), device='cpu'):
    """Generate 3D animation of a static volume by making circular orbit around object

    Args:
        volume (tensor): volume to preview of shape (width, height, depth) or
            (width, height, depth, num_channels) for multi-channel measurement
        grid (SphericalGrid): grid where volume is defined
        positions (int): number of positions in orbit
        shape (tuple[int]): shape of output images

    Returns:
        tensor: stack of images containing rotating preview of volume with shape
            (positions, *shape) if volume is single channel, or
            (positions, *shape, num_channels) if multiple channels

    Usage
    """

    if not volume.ndim in (3, 4):
        raise ValueError(f"Invalid shape for volume: {tuple(volume.shape)}")

    # if volume.ndim == 4:
    #     g = SphericalGrid(shape=volume.shape[:-1])
    # elif volume.ndim == 3:
    #     g = SphericalGrid(shape=volume.shape)
    # else:
    #     raise ValueError(f"Invalid shape for volume: {tuple(volume.shape)}")

    # rotate volume instead of creating many views
    offsets = tr.div(tr.arange(positions) * grid.shape.a, positions, rounding_mode='floor')

    geom = ConeRectGeom(shape, pos=(4 * grid.size.r[1], 0, 1), fov=(30, 30))
    op = Operator(grid, geom)

    # if multiple channels, process each separately
    if volume.ndim == 4:
        rotvol = tr.empty((positions, *grid.shape, 3))
        for i, offset in enumerate(offsets):
            rotvol[i] = tr.roll(volume, (0, 0, int(offset), 0), dims=(0, 1, 2, 3))
        results = []
        for chan in tr.moveaxis(rotvol, -1, 0):
            results.append(op(chan))
        return tr.stack(results, axis=-1)
    else:
        rotvol = tr.empty((positions, *grid.shape))
        for i, offset in enumerate(offsets):
            rotvol[i] = tr.roll(volume, (0, 0, int(offset)), dims=(0, 1, 2))
        return op(rotvol)
