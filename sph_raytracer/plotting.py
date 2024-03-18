#!/usr/bin/env python3

from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np
import torch as tr
from itertools import chain

from .geometry import SphericalGrid, ConeRectGeom
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


def image_stack(images, ax=None, polar=False, colorbar=False, **kwargs):
    """Animate a stack of images

    Args:
        images (ndarray or tensor): array of shape (num_images, width, height)
        ax (matplotlib Axes, optional): existing Axes object to use
        polar (bool): polar plot
        colorbar (bool): include a colorbar
        **kwargs: arguments to pass to plot

    Returns:
        Animation
    """
    if ax is None:
        fig = plt.figure(figsize=(3, 3))
        ax = fig.add_subplot(polar=polar)

    # detach from GPU if necessary
    if isinstance(images, tr.Tensor):
        images = images.detach().cpu().numpy()

    if polar:
        theta_lin = np.linspace(0, 2*np.pi, 100)
        # r_lin = np.logspace(np.log10(3), np.log10(25), 100)
        r_lin = np.linspace(3, 25, 100)
        theta, r = np.meshgrid(theta_lin, r_lin)
        imshow = lambda img, **k: ax.pcolormesh(theta, r, img, **k)
    else:
        imshow = ax.imshow

    vmin, vmax = images.min(), images.max()
    # artists = [[ax.imshow(im, animated=True)] for im in images]
    artists = [[imshow(im, animated=True, vmin=vmin, vmax=vmax, **kwargs)] for im in images]

    if colorbar:
        ax_col = ax.twinx()
        ax_col.tick_params(which="both", right=False, labelright=False)
        plt.colorbar(artists[0][0], ax=ax_col)

    return animation.ArtistAnimation(ax.figure, artists, interval=200)

def preview3d(volume, positions=20):
    """Generate 3D animation of a static volume by making circular orbit around object

    Args:
        volume (tensor): 3D tensor to preview
        positions (int): number of positions in orbit
    """

    g = SphericalGrid(shape=volume.shape)
    # rotate volume instead of creating many views
    rotvol = tr.empty((positions, *volume.shape))
    offsets = tr.div(tr.arange(positions) * g.shape.a, positions, rounding_mode='floor')
    for i, offset in enumerate(offsets):
        rotvol[i] = tr.roll(volume, (0, 0, int(offset)), dims=(0, 1, 2))

    op = Operator(g, ConeRectGeom((256, 256), pos=(4, 0, 1), fov=(30, 30)))

    return op(rotvol)
