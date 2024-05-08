"""Raytracer geometries

This modules contains classes for fully specifying the geometry of a tomographic operator.

SphericalGrid defines the shape and extent of the volume being raytraced, and ViewGeom (and its children)
define the shape, position, and orientation of the detector for each measurement.

The user may fully specify pixel lines-of-sight of custom detector with ViewGeom, or can use ConeCircGeom/ConeRectGeom
for a cone-beam detector with known FOV and uniform pixel pitch.
"""

from collections import namedtuple
import math
import torch as tr

__all__ = ['SphericalGrid', 'ConeRectGeom', 'ConeCircGeom',
           'ViewGeomCollection', 'ViewGeom', 'ParallelGeom'
           ]

Size = namedtuple('Size', ['r', 'e', 'a'])
Shape = namedtuple('Shape', ['r', 'e', 'a'])

FTYPE = tr.float64

class SphericalGrid:
    r"""Spherical grid information

    This class specifies the physical geometry of the volume being raytraced.

    The grid may be specified either by providing a shape and size of the grid,
    or by manually specifying the locations of all voxels.

    Args:
        size_r (tuple[float]): Radial extent of grid (r_min, r_max) with units of distance.
        size_e (tuple[float]): Elevational extent of grid (e_min, e_max) with units of radians.
        size_a (tuple[float]): Azimuthal extent of grid (e_min, e_max) with units of radians.
        shape (tuple[int]): shape of spherical grid (N rad. bins, N elev. bins, N az. bins)
        spacing (str): if `size` and `shape` given, space the radial bins linearly (spacing='lin')
            or logarithmically (spacing='log')
        rs_b (ndarray, optional): manually specify radial shell boundaries.
        phis_b (ndarray, optional): manually specify elevation cone boundaries
            in radians [0,π] (measured from +Z axis).
        thetas_b (ndarray, optional): manually specify azimuth plane boundaries
            in radians [-π,π] (measured from +X axis)

    Attributes:
        shape (tuple[int])
        rs (tensor[float]): radial bin centers
        phis (tensor[float]): elevation bin centers
        thetas (tensor[float]): azimuth bin centers
        rs_b (tensor[float])
        phis_b (tensor[float])
        thetas_b (tensor[float])
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
            self, size_r=(0, 1), size_e=(0, tr.pi), size_a=(-tr.pi, tr.pi), shape=(50, 50, 50), spacing='lin',
            rs_b=None, phis_b=None, thetas_b=None):
        size = Size(size_r, size_e, size_a)
        shape = Shape(*shape)

        # infer shape and size if grid is manually specified
        if (rs_b is not None) and (phis_b is not None) and (thetas_b is not None):
            shape = Shape(len(rs_b) - 1, len(phis_b) - 1, len(thetas_b) - 1)
            size = Size((min(rs_b), max(rs_b)), (min(phis_b), max(phis_b)), (min(thetas_b), max(thetas_b)))

            # enforce float64 dtype
            rs_b, phis_b, thetas_b = [tr.asarray(x, dtype=tr.float64) for x in (rs_b, phis_b, thetas_b)]
            rs, phis, thetas = [(x[1:] + x[:-1]) / 2 for x in (rs_b, phis_b, thetas_b)]

        # otherwise compute grid
        elif (shape is not None) and (size is not None):
            if spacing == 'log':
                rs_b = tr.logspace(math.log10(size.r[0]), math.log10(size.r[1]), shape.r + 1, dtype=tr.float64)
                rs = tr.sqrt(rs_b[1:] * rs_b[:-1])
            elif spacing == 'lin':
                rs_b = tr.linspace(size.r[0], size.r[1], shape.r + 1, dtype=tr.float64)
                rs = (rs_b[1:] + rs_b[:-1]) / 2
            else:
                raise ValueError("Invalid value for spacing")
            phis_b = tr.linspace(size.e[0], size.e[1], shape.e + 1, dtype=tr.float64)
            thetas_b = tr.linspace(size.a[0], size.a[1], shape.a + 1, dtype=tr.float64)
            phis = (phis_b[1:] + phis_b[:-1]) / 2
            thetas = (thetas_b[1:] + thetas_b[:-1]) / 2

        else:
            raise ValueError("Must specify either shape or (rs, phis, thetas)")


        self.size = size
        self.shape = shape
        self.rs_b, self.phis_b, self.thetas_b = rs_b, phis_b, thetas_b
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
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        return artist


# ----- Viewing Geometry -----

# wireframe segment
Segment = namedtuple('Segment', ['color', 'thickness', 'start', 'end'])

class ViewGeom:
    """Custom sensor with arbitrary ray placement.

    Create a custom viewing geometry by specifying the start positions (i.e. absolute pixel locations)
    and ray direction (i.e. pixel LOSs) for every pixel.  The pixels need not be in a grid or colocated in space.

    The detector may be any shape as long as the last dimension has length 3.  The shape of the detector controls the shape
    of images returned by the raytracer (`Operator`)

    Args:
        ray_starts (tensor): Pixel location array of shape (..., 3)
        rays (tensor): Pixel LOS array of shape (..., 3)

    Attributes:
        ray_starts (tensor):
        rays (tensor):
        shape (tuple): Shape of the detector (excluding last dimension of provided rays)

    Usage:

    """

    def __init__(self, ray_starts, rays):
        self.ray_starts = tr.asarray(ray_starts, dtype=FTYPE)
        self.rays = tr.asarray(rays, dtype=FTYPE)
        self.rays /= tr.linalg.norm(self.rays, axis=-1)[..., None]
        self.shape = self.rays.shape[:-1]

    def __add__(self, other):
        if other == 0 or other == None:
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

    @property
    def _wireframe(self):
        """(segments, widths, colors): Wireframe for 3D visualization"""
        # draw FOV corners

        ray_ends = (
            self.ray_starts +
            self.rays * 2 * tr.linalg.norm(self.ray_starts, dim=-1)[..., None]
        )
        segments = tr.stack((self.ray_starts.reshape(-1, 3), ray_ends.reshape(-1, 3)), dim=1)

        return [[segments, tr.ones(len(segments)), ['black'] * len(segments)]]


    def plot(self, ax=None):
        """Generate Matplotlib wireframe plot for this object

        Returns:
            matplotlib Axes
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Line3DCollection

        if ax is None:
            fig = plt.figure(figsize=(3, 3))
            ax = fig.add_subplot(projection='3d', computed_zorder=False)

        segments, widths, colors = wireframe[0]
        lc = Line3DCollection(segments, linewidths=widths, colors=colors)
        ax.add_collection(lc)

        # limits and labels
        lim = tr.abs(self.ray_starts).max()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim3d([-lim, lim])
        ax.set_ylim3d([-lim, lim])
        ax.set_zlim3d([-lim, lim])

        return ax


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

    def plot(self, ax=None):
        """Generate Matplotlib wireframe plot for this object

        Returns:
            matplotlib Axes
        """
        import matplotlib.pyplot as plt
        from matplotlib import animation
        from mpl_toolkits.mplot3d.art3d import Line3DCollection

        if ax is None:
            fig = plt.figure(figsize=(3, 3))
            ax = fig.add_subplot(projection='3d', computed_zorder=False)

        wireframe = self._wireframe

        lc = Line3DCollection([])
        ax.add_collection(lc)

        def update(num):
            segments, widths, colors = wireframe[num]
            lc.set_segments(segments)
            lc.set_linewidth(widths)
            lc.set_colors(colors)
            return lc,
        self._update = update
        update(0)
        # limits and labels
        # lim = max(tr.linalg.norm(self.geom.ray_starts, dim=-1))
        lim = tr.abs(self.ray_starts).max()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim3d([-lim, lim])
        ax.set_ylim3d([-lim, lim])
        ax.set_zlim3d([-lim, lim])

        N = len(wireframe)
        return animation.FuncAnimation(ax.figure, update, N, interval=3000/N, blit=False)


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
        + u[None, None, :] * tr.linspace(ulim, -ulim, self.shape[0])[:, None, None]
        + v[None, None, :] * tr.linspace(-vlim, vlim, self.shape[1])[None, :, None]
        ).reshape((*self.shape, 3))
        rays /= tr.linalg.norm(rays, axis=-1)[..., None]
        return rays

    @property
    def ray_starts(self):
        """Start position of each ray. Shape (1, 3)"""
        return self.pos[None, None, :]

    def __repr__(self):
        string = f"""{self.__class__.__name__}(
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
        return [[segments, tr.ones(len(segments)), ['black'] * len(segments)]]


class ConeCircGeom(ConeRectGeom):
    """Circular sensor with cone beam geometry

    Args:
        shape (tuple[int]): detector shape (npix_r, npix_theta)
        pos (tuple[float]): XYZ position of detector
        lookdir (tuple[float]): detector pointing direction
        updir (tuple[float]): direction of detector +Y
        fov (float): detector field of view
    """

    def __init__(self, *args, fov=45, spacing='lin', **kwargs):
        super().__init__(*args, fov=fov, **kwargs)

        # build r, theta grid
        # https://math.stackexchange.com/questions/73237/parametric-equation-of-a-circle-in-3d-space
        if spacing == 'lin':
            self.r = tr.linspace(0, tr.tan(tr.deg2rad(self.fov / 2)), self.shape[0])
        elif spacing == 'log':
            self.r = tr.logspace(0, tr.tan(tr.deg2rad(self.fov / 2)), self.shape[0])
        else:
            raise ValueError(f"Invalid spacing {spacing}")

        self.theta = tr.linspace(0, 2 * tr.pi, self.shape[1])

    @property
    def rays(self):
        """Ray unit vectors. Shape (*shape, 3)"""
        u = tr.cross(self.lookdir, self.updir)
        v = self.updir

        rays = (
            self.lookdir[None, None, :]
            + self.r[:, None, None] * tr.cos(self.theta[None, :, None]) * u[None, None, :]
            + self.r[:, None, None] * tr.sin(self.theta[None, :, None]) * v[None, None, :]
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
        return [[segments, tr.ones(len(segments)), ['black'] * len(segments)]]


class ParallelGeom(ViewGeom):
    """Rectangular parallel beam sensor

    Args:
        shape (tuple[int]): detector shape (npix_x, npix_y)
        pos (tuple[float]): XYZ position of detector center
        lookdir (tuple[float]): detector pointing direction
        updir (tuple[float]): direction of detector +Y
        size (tuple[float]): size of detector in distance units (width, height)
    """

    def __init__(self, shape, pos, lookdir=None, updir=None, size=(1, 1)):
        pos = tr.asarray(pos, dtype=FTYPE)
        if lookdir is None:
            lookdir = -pos
        else:
            lookdir = tr.asarray(lookdir, dtype=FTYPE)
        if updir is None:
            updir = tr.cross(lookdir, tr.asarray((0, 0, 1), dtype=FTYPE))
        else:
            updir = tr.asarray(updir, dtype=FTYPE)
        lookdir /= tr.linalg.norm(lookdir, axis=-1)
        updir /= tr.linalg.norm(updir, axis=-1)


        u = tr.cross(lookdir, updir)
        v = updir

        # handle case with single LOS
        ulim = size[0]/2 if shape[0] > 1 else 0
        vlim = size[1]/2 if shape[1] > 1 else 0
        self._u_arr = u[None, None, :] * tr.linspace(ulim, -ulim, shape[0])[:, None, None]
        self._v_arr = v[None, None, :] * tr.linspace(-vlim, vlim, shape[1])[None, :, None]

        self.shape = shape
        self.pos = pos
        self.lookdir = lookdir
        self.updir = updir
        self.size = size

    @property
    def rays(self):
        """Ray unit vectors (1, 1, 3)"""
        return self.lookdir[None, None, :]

    @property
    def ray_starts(self):
        """Start position of each ray. Shape (*shape, 3)"""
        return (self.pos[None, None, :] + self._u_arr + self._v_arr).reshape((*self.shape, 3))

    def __repr__(self):
        string = f"""ParallelGeom(
            shape={self.shape}
            pos={self.pos.tolist()},
            lookdir={self.lookdir.tolist()},
        )"""
        from inspect import cleandoc
        return cleandoc(string)

    @property
    def _wireframe(self):
        """(segments, widths, colors): Wireframe for 3D visualization"""
        # draw FOV corners

        corners_start = self.ray_starts[(-1, -1, 0, 0), (0, -1, -1, 0)]
        corners_end = (
            corners_start + self.lookdir[None, :] * 2*tr.linalg.norm(self.pos)
        )

        cone_lines = tr.stack((corners_start, corners_end), dim=1)
        plane_start_lines = tr.stack((corners_start, corners_start.roll(-1, dims=0)), dim=1)
        plane_end_lines = tr.stack((corners_end, corners_end.roll(-1, dims=0)), dim=1)

        segments = tr.concat((cone_lines, plane_start_lines, plane_end_lines))
        return [[segments, tr.ones(len(segments)), ['black'] * len(segments)]]