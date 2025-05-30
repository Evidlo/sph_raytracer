#!/usr/bin/env python3

import copy
from matplotlib import animation
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter
import numpy as np
import torch as tr
from itertools import chain, repeat
from collections.abc import Iterable

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

def image_stack(images, geom=None, ax=None, colorbar=False, polar=None, **kwargs):
    """Animate a stack of images

    Args:
        images (ndarray or tensor): array of shape (num_images, width, height)
            for an animated sequence or (width, height) for a single image
        geom (ConeRectGeom, ConeCircGeom or ViewGeomCollection): view geometry
            for labelling FOV.  If ViewGeomCollection, all geometries should be
            of the same type
        ax (matplotlib Axes, optional): existing Axes object to use
        colorbar (bool): include a colorbar
        polar (bool): override polar plot detection
        **kwargs: arguments to pass to plot

    Returns:
        matplotlib.animation.ArtistAnimation
    """
    # FIXME: function doesn't work with channels
    ispolar = lambda g: isinstance(g, ConeCircGeom)
    isiterable = lambda g: isinstance(g, (ViewGeomCollection, Iterable))
    if polar is None:
        polar = ispolar(geom) or (isiterable(geom) and ispolar(geom[0]))
    if ax is None:
        fig = plt.figure(figsize=(3, 3))
        ax = fig.add_subplot(polar=polar)
    else:
        fig = ax.figure

    # detach from GPU if necessary
    if isinstance(images, tr.Tensor):
        images = images.detach().cpu().numpy()

    deg_format = EngFormatter(unit=u"°", sep="")

    if polar:
        def imshow(img, geom, **kwargs):
            # r_lin = np.logspace(np.log10(3), np.log10(25), 100)
            if geom is not None:
                ax.yaxis.set_major_formatter(deg_format)
                fov = geom.fov
            else:
                fov = (0, 1)
            # FIXME: remove carruthers specific hack
            if hasattr(geom, 'sb'):
                r_lin = geom.sb.rad_ctrs / (geom.sb.npix / 2) * geom.fov[1]
                theta_lin = np.deg2rad(geom.sb.theta_ctrs)
            else:
                r_lin = np.linspace(fov[0]/2, fov[1]/2, images.shape[-2] + 1)
                theta_lin = np.linspace(0, 2*np.pi, images.shape[-1] + 1)
            # realign polar plot up direction
            # ax.set_theta_zero_location('N')
            theta, r = np.meshgrid(theta_lin, r_lin)
            ax.grid(alpha=.2)
            return ax.pcolormesh(theta, r, img, **kwargs)
    else:
        def imshow(img, geom, **kwargs):
            if geom is not None:
                extent = (-geom.fov[1]/2, geom.fov[1]/2, -geom.fov[0]/2, geom.fov[0]/2)
                ax.xaxis.set_major_formatter(deg_format)
                ax.yaxis.set_major_formatter(deg_format)
            else:
                extent = None
            return ax.imshow(img, extent=extent, **kwargs)

    if not {'vmin', 'vmax'} <= kwargs.keys():
        kwargs['vmin'], kwargs['vmax'] = np.nanmin(images), np.nanmax(images)
    if images.ndim == 3:
        # use same ViewGeom for all images if a ViewGeomCollection not provided
        geom = geom if isiterable(geom) else repeat(geom)
        artists = [[imshow(im, g, animated=True, **kwargs)] for im, g in zip(images, geom)]
        result = animation.ArtistAnimation(ax.figure, artists, interval=200)
    elif images.ndim == 2:
        artists = [[imshow(images, geom, **kwargs)]]
        # result = artists[0][0]
        result = fig
    else:
        raise ValueError("Invalid images shape")

    if colorbar:
        ax.figure.colorbar(artists[0][0], pad=.1)

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


def sph2cart(rea):
    """Convert spherical coordinates to cartesian coordinates

    Args:
        spherical (tuple): spherical coordinates (radius, elevation, azimuth),
            where elevation is measured from Z-axis in radians [0, ℼ] and
            azimuth is measured from X-axis in radians [-ℼ, ℼ]

    Returns:
        cartesian (tuple): cartesian coordinates (x, y, z)
    """
    r, e, a = np.moveaxis(rea, -1, 0)

    xyz = np.empty_like(rea)

    pre_selector = ((slice(None),) * xyz.ndim)[:-1]

    xyz[(*pre_selector, 0)] = r * np.sin(e) * np.cos(a)
    xyz[(*pre_selector, 1)] = r * np.sin(e) * np.sin(a)
    xyz[(*pre_selector, 2)] = r * np.cos(e)

    return xyz


def preview3d(volume, grid, shape=(256, 256), orbit=True, elev=60, azim=0, device='cpu'):
    """Generate 3D animation of a static volume by making circular orbit around object

    The number of frames is equal to the number of time bins (dynamic volume), or number
    of azimuth bins (static volume)

    This function cheats by reusing the same view geometry for every frame and just
    rotating the volume by one azimuth bin, greatly reducing memory and CPU overhead.

    Args:
        volume (tensor): volume to preview of shape (width, height, depth) or
            (width, height, depth, num_channels) for multi-channel measurement
        grid (SphericalGrid): grid where volume is defined
        shape (tuple[int]): shape of output images
        orbit (bool): orbit around volume (default True)
        elev (float): initial camera elevation (degrees, default 60°)
        azim (float): initial camera azimuth (degrees, default 0°)
        device (str): PyTorch device to use for computing preview (default 'cpu')

    Returns:
        tensor: stack of images containing rotating preview of volume with shape
            (grid.shape.a, *shape) if volume is static single channel, or
            (grid.shape.a, *shape, num_channels) if static multiple channels, or
            (grid.shape.t, *shape, num_channels) if dynamic single channel, or
            (grid.shape.t, *shape, num_channels) if dynamic multiple channels

    Example returned shapes:
        # dynamic multi channel (RGB)
        volume:   (20, 50, 50, 50, 3)
        grid:     (20, 50, 50, 50)
        returned: (20, 256, 256, 3)

        # static multi channel (RGB)
        volume:   (50, 50, 50, 3)
        grid:     (50, 50, 50)
        returned: (50, 256, 256, 3)

        # dynamic single channel
        volume:   (20, 50, 50, 50)
        grid:     (20, 50, 50, 50)
        returned: (20, 256, 256)

        # static single channel
        volume:   (50, 50, 50)
        grid:     (50, 50, 50)
        returned: (50, 256, 256)
    """

    if not volume.ndim in (3, 4, 5):
        raise ValueError(f"Invalid shape for volume: {tuple(volume.shape)}")
    if not (volume.ndim - len(grid.shape)) <= 1:
        # volume may have 1 more dimension than grid if channels are present
        raise ValueError("volume/grid shape mismatch")

    # rotate volume instead of creating many views
    # offsets = tr.div(tr.arange(positions) * grid.shape.a, positions, rounding_mode='floor')
    offsets = range(grid.shape.t if grid.dynamic else grid.shape.a)

    # offset view by 1/2 azimuth voxel to avoid visual artifacts
    pos = sph2cart((
        4 * grid.size.r[1],
        np.deg2rad(elev),
        0.125 * 2 * np.pi / grid.shape.a + np.deg2rad(azim)
    ))
    geom = ConeRectGeom(shape, pos=pos, fov=(30, 30))
    # geom = ConeRectGeom(shape, pos=(4 * grid.size.r[1], halfaz, 1 * grid.size.r[1]), fov=(30, 30))
    # FIXME: flatten this too?
    op = Operator(grid, geom)

    # if multiple channels, process each separately
    if volume.ndim == len(grid.shape) + 1:
        rotvol = tr.empty((grid.shape.a, *grid.shape[-3:], volume.shape[-1]))
        for i, offset in enumerate(offsets):
            vol = volume[offset] if grid.dynamic else volume
            # roll azimuth dimension
            rotvol[i] = tr.roll(vol, offset if orbit else 0, dims=[-2])
        results = []
        for chan in tr.moveaxis(rotvol, -1, 0):
            results.append(op(chan))
        return tr.stack(results, axis=-1)
    # only a single channel
    elif volume.ndim == len(grid.shape):
        rotvol = tr.empty((len(offsets), *grid.shape[-3:]))
        for i, offset in enumerate(offsets):
            vol = volume[offset] if grid.dynamic else volume
            # roll azimuth dimension
            rotvol[i] = tr.roll(vol, offset if orbit else 0, dims=[-1])
        return op(rotvol)
