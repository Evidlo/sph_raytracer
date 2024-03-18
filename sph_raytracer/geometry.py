#!/usr/bin/env python3

from collections import namedtuple
import math
import torch as tr

__all__ = ['SphericalGrid', 'ConeRectGeom', 'ConeCircGeom', 'ViewGeomCollection', 'ViewGeom']

Size = namedtuple('Size', ['r', 'e', 'a'])
Shape = namedtuple('Shape', ['r', 'e', 'a'])

FTYPE = tr.float64

class SphericalGrid:
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
        SphericalGrid(((3, 25), (0, tr.pi), (-tr.pi, tr.pi)), (50, 50, 50))
        SphericalGrid(
            rs=tr.linspace(3, 25, 51),
            phis=tr.linspace(0, tr.pi, 51),
            thetas=tr.linspace(-tr.pi, tr.pi, 51)
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

            # enforce flaot64 dtype
            rs, phis, thetas = [tr.asarray(x, dtype=tr.float64) for x in (rs, phis, thetas)]

        # otherwise compute grid
        elif (shape is not None) and (size is not None):
            if spacing == 'log':
                rs = tr.logspace(math.log10(size.r[0]), math.log10(size.r[1]), shape.r + 1)
            elif spacing == 'lin':
                rs = tr.linspace(size.r[0], size.r[1], shape.r + 1)
            else:
                raise ValueError("Invalid value for spacing")
            phis = tr.linspace(size.e[0], size.e[1], shape.e + 1, dtype=tr.float64)
            thetas = tr.linspace(size.a[0], size.a[1], shape.a + 1, dtype=tr.float64)

        else:
            raise ValueError("Must specify either shape or (rs, phis, thetas)")

        self.size = size
        self.shape = shape
        self.rs, self.phis, self.thetas = rs, phis, thetas

    def __repr__(self):
        s, sh = self.size, self.shape
        size = f"(({s[0][0]:.1f}, {s[0][1]:.1f}), ({s[1][0]:.1f}, {s[1][1]:.1f}), ({s[2][0]:.1f}, {s[2][1]:.1f}))"
        shape = f"({sh[0]}, {sh[1]}, {sh[2]})"
        string = f"""SphericalGrid(
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
        u = tr.linspace(0, 2 * tr.pi, 20)
        v = tr.linspace(0, tr.pi, 20)
        x = tr.outer(tr.cos(u), tr.sin(v)) * self.size[0][1]
        y = tr.outer(tr.sin(u), tr.sin(v)) * self.size[0][1]
        z = tr.outer(tr.ones_like(u), tr.cos(v)) * self.size[0][1]

        # Plot the surface
        artist = ax.plot_surface(x, y, z, zorder=-999)
        ax.set_aspect('equal')

        return artist


# ----- Viewing Geometry -----

# wireframe segment
Segment = namedtuple('Segment', ['color', 'thickness', 'start', 'end'])

class ViewGeom:
    """Custom sensor with arbitrary ray placement"""

    def __init__(self, ray_starts, rays):
        self.ray_starts = tr.asarray(ray_starts, dtype=FTYPE)
        self.rays = tr.asarray(rays, dtype=FTYPE)
        self.rays /= tr.linalg.norm(self.rays, axis=-1)[..., None]
        self.shape = self.rays.shape[:-1]

    def __add__(self, other):
        if other == 0:
            return ViewGeomCollection(self)
        if isinstance(other, ViewGeomCollection):
            other.geoms.append(self)
            return other
        else:
            return ViewGeomCollection(self, other)

    def __radd__(self, other):
        return self.__add__(other)

    def __repr__(self):
        cls = self.__class__.__name__
        string = f"""{cls}(
            shape={tuple(self.shape)}
        )"""
        from inspect import cleandoc
        return cleandoc(string)


class ViewGeomCollection(ViewGeom):
    """Set of viewing geometries

    Args:
        *geoms (ViewGeom): ViewGeoms with same shape
    """
    def __init__(self, *geoms):
        if not all(g.shape == geoms[0].shape for g in geoms):
            raise ValueError("ViewGeoms must all have same shape")
        self.geoms = list(geoms)

    def __add__(self, other):
        if isinstance(other, ViewGeomCollection):
            self.geoms += other.geoms
            other.geoms += self.geoms
        else:
            self.geoms.append(other)
        return self

    def __radd__(self, other):
        return self.__add__(other)

    def __getitem__(self, ind):
        return self.geoms[ind]

    def __len__(self):
        return len(self.geoms)

    @property
    def shape(self):
        return (len(self.geoms), *self.geoms[0].shape)

    @property
    def rays(self):
        return tr.concat(tuple(g.rays[None, ...] for g in self.geoms))

    @property
    def ray_starts(self):
        return tr.concat(tuple(g.ray_starts[None, ...] for g in self.geoms))

    @property
    def _wireframe(self):
        """(segments, widths, colors): Wireframe for 3D visualization"""
        return sum([g._wireframe for g in self.geoms], [])


class ConeRectGeom(ViewGeom):
    """Rectangular sensor with cone beam geometry

    Args:
        shape (tuple[int]): detector shape (npix_x, npix_y)
        pos (tuple[float]): XYZ position of detector
        lookdir (tuple[float]): detector pointing direction
        updir (tuple[float]): direction of detector +Y
        fov (tuple[float]): detector field of view (fov_x, fov_y)
    """

    def __init__(self, shape, pos, lookdir=None, updir=None, fov=(45, 45)):
        pos = tr.asarray(pos, dtype=FTYPE)
        if lookdir is None:
            lookdir = -pos
        else:
            lookdir = tr.asarray(lookdir, dtype=FTYPE)
        if updir is None:
            updir = tr.cross(lookdir, tr.asarray((0, 0, 1), dtype=FTYPE))
        else:
            updir = tr.asarray(updir, dtype=FTYPE)
        fov = tr.asarray(fov, dtype=FTYPE)
        lookdir /= tr.linalg.norm(lookdir, axis=-1)
        updir /= tr.linalg.norm(updir, axis=-1)

        self.shape = shape
        self.pos = pos
        self.lookdir = lookdir
        self.updir = updir
        self.fov = fov

    @property
    def rays(self):
        """Ray unit vectors (*shape, 3)"""
        u = tr.cross(self.lookdir, self.updir)
        v = self.updir

        # handle case with single LOS
        ulim = tr.tan(tr.deg2rad(self.fov[0] / 2)) if self.shape[0] > 1 else 0
        vlim = tr.tan(tr.deg2rad(self.fov[1] / 2)) if self.shape[1] > 1 else 0
        rays = (
        self.lookdir[None, None, :]
        + u[None, None, :] * tr.linspace(ulim, -ulim, self.shape[0])[None, :, None]
        + v[None, None, :] * tr.linspace(vlim, -vlim, self.shape[1])[:, None, None]
        ).reshape((*self.shape, 3))
        rays /= tr.linalg.norm(rays, axis=-1)[..., None]
        return rays

    @property
    def ray_starts(self):
        """Start position of each ray. Shape (1, 3)"""
        return self.pos[None, None, :]

    def __repr__(self):
        string = f"""ConeRectGeom(
            shape={self.shape}
            pos={self.pos.tolist()},
            lookdir={self.lookdir.tolist()},
            fov={self.fov.tolist()}
        )"""
        from inspect import cleandoc
        return cleandoc(string)

    @property
    def _wireframe(self):
        """(segments, widths, colors): Wireframe for 3D visualization"""
        # draw FOV corners

        corners = self.rays[(-1, -1, 0, 0), (0, -1, -1, 0)]
        corners *= tr.linalg.norm(self.pos)

        cone_lines = tr.stack((self.pos.broadcast_to(corners.shape), corners), dim=1)
        plane_lines = tr.stack((corners, corners.roll(-1, dims=0)), dim=1)

        segments = tr.concat((cone_lines, plane_lines))
        return [[segments, tr.ones(len(segments)), ['dimgray'] * len(segments)]]


class ConeCircGeom(ConeRectGeom):
    """Circular sensor with cone beam geometry

    Args:
        shape (tuple[int]): detector shape (npix_r, npix_theta)
        pos (tuple[float]): XYZ position of detector
        lookdir (tuple[float]): detector pointing direction
        updir (tuple[float]): direction of detector +Y
        fov (float): detector field of view
    """

    def __init__(self, *args, fov=45, **kwargs):
        super().__init__(*args, fov=fov, **kwargs)

    @property
    def rays(self):
        """Ray unit vectors. Shape (*shape, 3)"""
        u = tr.cross(self.lookdir, self.updir)
        v = self.updir

        # build r, theta grid
        # https://math.stackexchange.com/questions/73237/parametric-equation-of-a-circle-in-3d-space
        r = tr.linspace(0, tr.tan(tr.deg2rad(self.fov / 2)), self.shape[0])
        theta = tr.linspace(0, 2 * tr.pi, self.shape[1])
        rays = (
            self.lookdir[None, None, :]
            + r[:, None, None] * tr.cos(theta[None, :, None]) * u[None, None, :]
            + r[:, None, None] * tr.sin(theta[None, :, None]) * v[None, None, :]
        )
        rays /= tr.linalg.norm(rays, axis=-1)[..., None]
        return rays

    @property
    def _wireframe(self):
        """(segments, widths, colors): Wireframe for 3D visualization"""

        outer = self.rays[-1]
        outer *= tr.linalg.norm(self.pos)

        # sample up to 5 points on outer edge
        sampling = math.ceil(len(outer) / 4)
        cone_lines = tr.stack((self.pos.broadcast_to(outer[::sampling].shape), outer[::sampling]), dim=1)
        # endplane
        plane_lines = tr.stack((outer, outer.roll(-1, dims=0)), dim=1)

        segments = tr.concat((cone_lines, plane_lines))
        return [[segments, tr.ones(len(segments)), ['dimgray'] * len(segments)]]