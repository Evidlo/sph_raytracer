#!/usr/bin/env python3

from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np
import torch as tr
from itertools import chain

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


def image_stack(images, ax=None, polar=False, **kwargs):
    """Animate a stack of images

    Args:
        images (ndarray or tensor): array of shape (num_images, width, height)
        ax (matplotlib Axes, optional): existing Axes object to use
        polar (bool): polar plot
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

    # artists = [[ax.imshow(im, animated=True)] for im in images]
    artists = [[imshow(im, animated=True, **kwargs)] for im in images]

    return animation.ArtistAnimation(ax.figure, artists, interval=200)