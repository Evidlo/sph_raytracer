#!/usr/bin/env python3

from collections import namedtuple
import math
import torch as tr

Size = namedtuple('Size', ['r', 'e', 'a'])
Shape = namedtuple('Shape', ['r', 'e', 'a'])

class SphericalVol:
    r"""Spherical grid information

    Args:
        size (tuple[tuple[float]]): Tuple of ranges for each dimension
            ((r_min, r_max), (e_min, e_max), (a_min, a_max))
        shape (tuple[int]): shape of spherical grid (r bins, elev bins, az bins)
        spacing (str): radial shell spacing
        rs (ndarray, optional): manually specify radial shell locations.
        phis (ndarray, optional): manually specify elevation cone locations
            in radians [0,π] (measured from +Z axis).
        thetas (ndarray, optional): manually specify azimuth plane locations
            in radians [-π,π] (measured from +X axis)

    Attributes:
        shape (tuple[int])
        rs (ndarray[float])
        phis (ndarray[float])
        thetas (ndarray[float])
        size: (tuple[tuple[float]])

    Usage:
        SphericalVol(((3, 25), (0, tr.pi), (-tr.pi, tr.pi)), (50, 50, 50))
        SphericalVol(
            rs=tr.linspace(3, 25, 10),
            phis=tr.linspace(0, tr.pi, 10),
            thetas=tr.linspace(-tr.pi, tr.pi, 10)
        )

    Below is an illustration of where grid indices are located relative to
    voxel indices for a volume of shape (2, 2, 4)

             .....

          Radial (r)              Elevation (phi)           Azimuth (theta)
          ----------              ---------------           ---------------
                                         Z↑                        Y↑
   ..........* 2 *...........
   ........*       *.........
   ......*           *.......       0.........0             ..4         3
   ....*     **1**     *.....        \..-1.../              ...\   3   /
   ...*    *       *    *....         \...../               ....\     /
   ..*    *   *0*   *    *...       0  \.../  0             .....\   /  2
   ..*   *   *...*   *   *...           \./                 ..5...\ /
   ..*   *   *-1.* 0 * 1 *.2.    1-------+-------1  X→      .......+-------2   X→
   ..*   *   *...*   *   *...           /.\                 .-1.../ \
   ..*    *   ***   *    *...       1  /...\  1             ...../   \  1
   ...*    *       *    *....         /.....\               ..../     \
   ....*     *****     *.....        /...2...\              .../   0   \
   ......*           *.......       2.........2             ..0         1
   ........*       *.........
   ..........*****...........

                              ....
                              .... out of bounds voxels
                              ....

    """

    def __init__(
            self, size=((0, 1), (0, tr.pi), (-tr.pi, tr.pi)), shape=(50, 50, 50), spacing='lin',
            rs=None, phis=None, thetas=None):
        size = Size(*size)
        shape = Shape(*shape)

        # infer shape and size if grid is manually specified
        if (rs is not None) and (phis is not None) and (thetas is not None):
            shape = (len(rs) - 1, len(phis) - 1, len(thetas) - 1)
            size = ((min(rs), max(rs)), (min(phis), max(phis)), (min(thetas), max(thetas)))

        # otherwise compute grid
        elif (shape is not None) and (size is not None):
            if spacing == 'log':
                rs = tr.logspace(math.log10(size.r[0]), math.log10(size.r[1]), shape.r + 1)
            elif spacing == 'lin':
                rs = tr.linspace(size.r[0], size.r[1], shape.r + 1)
            else:
                raise ValueError("Invalid value for spacing")
            phis = tr.linspace(size.e[0], size.e[1], shape.e + 1)
            thetas = tr.linspace(size.a[0], size.a[1], shape.a + 1)

        else:
            raise ValueError("Must specify either shape or (rs, phis, thetas)")

        self.size = size
        self.shape = shape
        self.rs, self.phis, self.thetas = rs, phis, thetas

    def __repr__(self):
        s, sh = self.size, self.shape
        size = f"(({s[0][0]:.1f}, {s[0][1]:.1f}), ({s[1][0]:.1f}, {s[1][1]:.1f}), ({s[2][0]:.1f}, {s[2][1]:.1f}))"
        shape = f"({sh[0]}, {sh[1]}, {sh[2]})"
        string = f"""SphericalVol(
            size={size},
            shape={tuple(self.shape)}
        )"""
        from inspect import cleandoc
        return cleandoc(string)


    def plot(self, ax=None):
        """Generate Matplotlib wireframe plot for this object

        Args:
            ax (matplotlib Axes3D): existing matplotlib axis to use

        Returns
            matplotlib Axes3D
        """
        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.axes(projection='3d')
            ax.set_proj_type('persp')

        # Make data
        u = tr.linspace(0, 2 * tr.pi, 100)
        v = tr.linspace(0, tr.pi, 100)
        x = tr.outer(tr.cos(u), tr.sin(v)) * self.size[0][1]
        y = tr.outer(tr.sin(u), tr.sin(v)) * self.size[0][1]
        z = tr.outer(tr.ones_like(u), tr.cos(v)) * self.size[0][1]

        # Plot the surface
        ax.plot_surface(x, y, z)
        ax.set_aspect('equal')

        return ax


# ----- Viewing Geometry -----

# wireframe segment
Segment = namedtuple('Segment', ['color', 'thickness', 'start', 'end'])

class ViewGeom:
    """Custom sensor with arbitrary ray placement"""

    # def make2d(self, x):
    #     """Make a 1D input into a 2D tensor"""
    #     return tr.asarray(x, dtype=tr.float)[None, :]

    def __init__(self, pos, rays):
        pos = tr.asarray(pos, dtype=tr.float)
        rays = tr.asarray(rays, dtype=tr.float)
        rays /= tr.linalg.norm(rays, axis=-1)

    @property
    def shape(self):
        return self.rays.shape[:-1]

    # def __add__(self, other):
    #     return self

    # def __radd__(self, other):
    #     return self.__add__(other)

    def plot(self, ax=None):
        """Generate Matplotlib wireframe plot for this object

        Args:
            ax (matplotlib Axes3D): existing matplotlib axis to use

        Returns
            matplotlib Axes3D
        """
        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.axes(projection='3d')
            ax.set_proj_type('persp')

        for s in self._wireframe():
            # defining coordinates for the 2 points.
            x, y, z = zip(s.start, s.end)
            ax.plot3D(x, y, z, color=s.color)

        ax.set_aspect('equal')
        return ax

    def __repr__(self):
        string = f"""ViewGeom(
            pos={tuple(self.pos.tolist())},
            shape={tuple(self.shape)}
        )"""
        from inspect import cleandoc
        return cleandoc(string)

class ConeRectGeom(ViewGeom):
    """Rectangular sensor with fan/cone beam geometry"""

    def __init__(self, shape, pos, lookdir, updir=(0, 0, 1), fov=(45, 45)):
        # pos = self.make2d(pos)
        # lookdir = self.make2d(lookdir)
        # updir = self.make2d(updir)
        # fov = self.make2d(fov)
        pos = tr.asarray(pos, dtype=tr.float)
        lookdir = tr.asarray(lookdir, dtype=tr.float)
        updir = tr.asarray(updir, dtype=tr.float)
        fov = tr.asarray(fov, dtype=tr.float)
        lookdir /= tr.linalg.norm(lookdir, axis=-1)
        updir /= tr.linalg.norm(updir, axis=-1)

        # if not len(lookdir) == len(updir) == len(pos) == len(fov):
        #     raise ValueError("Input vectors should have equal shape")

        u = tr.cross(lookdir, updir)
        v = updir

        ulim = tr.tan(tr.deg2rad(fov[0] / 2))
        vlim = tr.tan(tr.deg2rad(fov[1] / 2))
        rays = (
        lookdir[None, None, :]
        + u[None, None, :] * tr.linspace(ulim, -ulim, shape[0])[:, None, None]
        + v[None, None, :] * tr.linspace(vlim, -vlim, shape[1])[None, :, None]
        ).reshape((*shape, 3))
        rays /= tr.linalg.norm(rays, axis=-1)[..., None]

        self.pos = pos
        self.lookdir = lookdir
        self.updir = updir
        self.fov = fov
        self.rays = rays


    def __repr__(self):
        string = f"""ConeRectGeom(
            shape={tuple(self.shape)}
            pos={tuple(self.pos.tolist())},
            lookdir={tuple(self.lookdir.tolist())},
            fov={tuple(self.fov.tolist())}
        )"""
        from inspect import cleandoc
        return cleandoc(string)

    def _wireframe(self):
        """Generate wireframe for 3D visualization"""
        # draw FOV corners

        corners = self.rays[(-1, -1, 0, 0), (0, -1, -1, 0)]
        corners *= tr.linalg.norm(self.pos)
        segments = [
            Segment('gray', 1, self.pos, c) for c in corners
        ]
        segments += [
            Segment('gray', 1, c1, c2) for c1, c2 in zip(corners, corners.roll(-1, dims=0))
        ]
        return segments
