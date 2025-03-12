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

StaticSize = namedtuple('Size', ['r', 'e', 'a'])
StaticShape = namedtuple('Shape', ['r', 'e', 'a'])
DynamicSize = namedtuple('Size', ['t', 'r', 'e', 'a'])
DynamicShape = namedtuple('Shape', ['t', 'r', 'e', 'a'])

FTYPE = tr.float64

class SphericalGrid:
    r"""Spherical grid information

    This class specifies the physical geometry of the volume being raytraced.

    The grid may be specified either by providing a shape and size of the grid,
    or by manually specifying the locations of all voxels.

    Args:
        size_t (tuple[float]): Temporal extent of grid (t_min, t_max) with units determined
            by `timeunit`
        size_r (tuple[float]): Radial extent of grid (r_min, r_max) with units of distance.
        size_e (tuple[float]): Elevational extent of grid (e_min, e_max) with units of radians.
        size_a (tuple[float]): Azimuthal extent of grid (e_min, e_max) with units of radians.
        shape (tuple[int]): shape of spherical grid (N_t, N_r, N_e, N_a)
            or (N_r, N_e, N_a) if static
        spacing (str): if `size` and `shape` given, space the radial bins linearly (spacing='lin')
            or logarithmically (spacing='log')
        t (ndarray, optional): manually specify temporal samples.
        r_b (ndarray, optional): manually specify radial shell boundaries.
        e_b (ndarray, optional): manually specify elevation cone boundaries
            in radians [0,π] (measured from +Z axis).
        a_b (ndarray, optional): manually specify azimuth plane boundaries
            in radians [-π,π] (measured from +X axis)
        timeunit (str, optional): unit of time values.  Default 's' (seconds)
            See https://numpy.org/devdocs/reference/arrays.datetime.html#arrays-dtypes-timeunits
            Useful when converting grid time bins back to e.g. np.datetime64

    Attributes:
        shape (tuple[int])
        t (tensor[int]): sample times
        r (tensor[float]): radial bin centers
        e (tensor[float]): elevation bin centers
        a (tensor[float]): azimuth bin centers
        r_b (tensor[float])
        e_b (tensor[float])
        a_b (tensor[float])
        size (tuple[tuple[float]])
        timeunit (str)
        dynamic (bool): whether grid is dynamic

    Usage:
        SphericalGrid((0, 1), (3, 25), (0, tr.pi), (-tr.pi, tr.pi), (10, 50, 50, 50))
        SphericalGrid(
            t=tr.linspace(0, 1, 10)
            r_b=tr.linspace(3, 25, 51),
            e_b=tr.linspace(0, tr.pi, 51),
            a_b=tr.linspace(-tr.pi, tr.pi, 51)
        )

    Below is an illustration of where grid indices are located relative to
    voxel indices for a volume of shape (T, 2, 2, 4)

             .....

          Radial (r)              Elevation (e)           Azimuth (a)
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
            self, shape=(50, 50, 50),
            size_t=(0, 1), size_r=(0, 1), size_e=(0, tr.pi), size_a=(-tr.pi, tr.pi),
            spacing='lin',
            t=None, r_b=None, e_b=None, a_b=None,
            timeunit='s',
            # FIXME: deprecated args
            rs_b=None, phis_b=None, thetas_b=None):

        # static volume
        if len(shape) == 3:
            size = StaticSize(size_r, size_e, size_a)
            shape = StaticShape(*shape[-3:])
            self.dynamic = False
        elif len(shape) == 4:
            size = DynamicSize(size_t, size_r, size_e, size_a)
            shape = DynamicShape(*shape)
            self.dynamic = True
        else:
            raise ValueError("shape must be 3D or 4D")

        # FIXME: deprecated arguments: phis_b, thetas_b, rs_b
        if (rs_b is not None) and (phis_b is not None) and (thetas_b is not None):
            r_b, e_b, a_b = rs_b, phis_b, thetas_b

        # infer shape and size if grid is manually specified
        if (r_b is not None) and (e_b is not None) and (a_b is not None):
            size_r = float(min(r_b)), float(max(r_b))
            size_e = float(min(e_b)), float(max(e_b))
            size_a = float(min(a_b)), float(max(a_b))

            if t is None:
                shape = StaticShape(len(r_b) - 1, len(e_b) - 1, len(a_b) - 1)
                size = StaticSize(size_r, size_e, size_a)
            else:
                size_t = float(min(t)), float(max(t))
                t = tr.asarray(t, dtype=tr.int64)
                shape = DynamicShape(len(t), len(r_b) - 1, len(e_b) - 1, len(a_b) - 1)
                size = DynamicSize(size_t, size_r, size_e, size_a)
                self.dynamic = True


            # enforce float64 dtype
            r_b, e_b, a_b = [tr.asarray(x, dtype=tr.float64) for x in (r_b, e_b, a_b)]
            r, e, a = [(x[1:] + x[:-1]) / 2 for x in (r_b, e_b, a_b)]

        # otherwise compute grid
        elif (shape is not None) and (size is not None):
            if len(shape) == 4:
                t = tr.linspace(size.t[0], size.t[1], shape.t, dtype=tr.float64)
            if spacing == 'log':
                r_b = tr.logspace(math.log10(size.r[0]), math.log10(size.r[1]), shape.r + 1, dtype=tr.float64)
                r = tr.sqrt(r_b[1:] * r_b[:-1])
            elif spacing == 'lin':
                r_b = tr.linspace(size.r[0], size.r[1], shape.r + 1, dtype=tr.float64)
                r = (r_b[1:] + r_b[:-1]) / 2
            else:
                raise ValueError("Invalid value for spacing")
            e_b = tr.linspace(size.e[0], size.e[1], shape.e + 1, dtype=tr.float64)
            a_b = tr.linspace(size.a[0], size.a[1], shape.a + 1, dtype=tr.float64)
            e = (e_b[1:] + e_b[:-1]) / 2
            a = (a_b[1:] + a_b[:-1]) / 2

        else:
            raise ValueError("Must specify either shape or (r, e, a)")


        self.size = size
        self.shape = shape
        self.spacing = spacing
        self.r_b, self.e_b, self.a_b = r_b, e_b, a_b
        self.t, self.r, self.e, self.a = t, r, e, a
        self.timeunit = timeunit

        # FIXME: deleteme, deprecated args
        self.rs_b, self.phis_b, self.thetas_b = r_b, e_b, a_b
        self.rs, self.phis, self.thetas = r, e, a

    def __repr__(self):
        string = f"{self.__class__.__name__}(\n"
        string += f'    shape={tuple(self.shape)},\n'
        for k, v in self.size._asdict().items():
            string += f'    size_{k}=({v[0]:.2f}, {v[1]:.2f}),\n'
        string += ')'
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
        x = tr.outer(tr.cos(u), tr.sin(v)) * self.size.r[1]
        y = tr.outer(tr.sin(u), tr.sin(v)) * self.size.r[1]
        z = tr.outer(tr.ones_like(u), tr.cos(v)) * self.size.r[1]

        # Plot the surface
        artist = ax.plot_surface(x, y, z, zorder=99)
        ax.set_aspect('equal')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        return artist

    def mesh(self):
        """tensor(float): Dense 3D or 4D (if dynamic) mesh of grid coordinates
        of shape (N_t, N_r, N_e, N_a, 4) dynamic or (N_r, N_e, N_a, 3) static
        """

        coords = ([self.t] if self.dynamic else []) + [self.r, self.e, self.a]
        return t.stack(t.meshgrid(coords, indexing='ij'), dim=-1)

    # @property
    # def shape(self):
    #     return len(self.r), len(self.e), len(self.a)


# ----- Viewing Geometry -----

# wireframe segment
Segment = namedtuple('Segment', ['color', 'thickness', 'start', 'end'])

class ViewGeom:
    """Custom sensor with arbitrary ray placement.

    Create a custom viewing geometry by specifying the start positions
    (i.e. absolute pixel locations) and ray direction (i.e. pixel LOSs)
    for every pixel in cartesian coordinates.  The pixels need not be
    in a spatial grid and may be placed arbitrarily

    The detector may be any shape as long as the last dimension has length 3.
    The shape of the detector controls the shape of images returned by the raytracer
    (`Operator`)

    Args:
        ray_starts (tensor): XYZ Pixel location array of shape (..., 3)
        rays (tensor): XYZ Pixel LOS array of shape (..., 3)

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
        ).reshape(-1, 3)
        ray_starts = self.ray_starts.reshape(-1, 3).broadcast_to(ray_ends.shape)
        segments = tr.stack((ray_starts, ray_ends), dim=1)

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

        segments, widths, colors = self._wireframe[0]
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
        if len(geoms) == 1 and hasattr(geoms[0], 'geoms'):
            self.geoms = geoms[0].geoms
        else:
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
    def pos(self):
        if all(hasattr(g, 'pos') for g in self.geoms):
            return tr.concat(tuple(g.pos[None, ...] for g in self.geoms))
        else:
            return None

    @property
    def _wireframe(self):
        """(segments, widths, colors): Wireframe for 3D visualization"""
        return sum([g._wireframe for g in self.geoms], [])

    def plot(self, ax=None):
        """Generate Matplotlib wireframe plot for this object

        Returns:
            matplotlib Animation
        """
        import matplotlib.pyplot as plt
        from matplotlib import animation
        from mpl_toolkits.mplot3d.art3d import Line3DCollection

        if ax is None:
            fig = plt.figure(figsize=(3, 3))
            ax = fig.add_subplot(projection='3d', computed_zorder=False)

        # draw path
        if (pos := self.pos) is not None:
            lc = Line3DCollection([])
            segments = tr.stack((pos[:-1], pos[1:]))
            lc.set_segments(segments)
            lc.set_linewidth(tr.ones(len(segments)))
            lc.set_colors(['gray'] * len(segments))
            ax.add_collection(lc)

        wireframe = self._wireframe
        lc = Line3DCollection([])
        ax.add_collection(lc)

        # update FOV wireframe on each frame
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

    Follows matplotlib image convention where pixel (0, 0) is top left
    corner of view and (-1, -1) is bottom right
    """

    def __init__(self, shape, pos, lookdir=None, updir=None, fov=(45, 45)):
        pos = tr.asarray(pos, dtype=FTYPE)
        if lookdir is None:
            lookdir = -pos
        else:
            lookdir = tr.asarray(lookdir, dtype=FTYPE)
        if updir is None:
            updir = tr.cross(lookdir, tr.asarray((0, 0, 1), dtype=FTYPE), dim=-1)
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
        u = tr.cross(self.lookdir, self.updir, dim=-1)
        v = self.updir

        # handle case with single LOS
        ulim = tr.tan(tr.deg2rad(self.fov[0] / 2)) if self.shape[0] > 1 else 0
        vlim = tr.tan(tr.deg2rad(self.fov[1] / 2)) if self.shape[1] > 1 else 0
        rays = (
        self.lookdir[None, None, :]
        + u[None, None, :] * tr.linspace(-ulim, ulim, self.shape[0])[:, None, None]
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

        corners = self.rays[(-1, -1, 0, 0), (0, -1, -1, 0)].clone()
        corners *= tr.linalg.norm(self.pos)
        corners += self.pos

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

        self.theta = tr.linspace(0, 2 * tr.pi, self.shape[1]) + tr.pi / 2

    @property
    def rays(self):
        """Ray unit vectors. Shape (*shape, 3)"""
        u = tr.cross(self.lookdir, self.updir, dim=-1)
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

        outer = self.rays[-1].clone()
        outer *= tr.linalg.norm(self.pos)
        outer += self.pos

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
            updir = tr.cross(lookdir, tr.asarray((0, 0, 1), dtype=FTYPE), dim=-1)
        else:
            updir = tr.asarray(updir, dtype=FTYPE)
        lookdir /= tr.linalg.norm(lookdir, axis=-1)
        updir /= tr.linalg.norm(updir, axis=-1)


        u = tr.cross(lookdir, updir, dim=-1)
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

        corners_start = self.ray_starts[(-1, -1, 0, 0), (0, -1, -1, 0)].clone()
        corners_end = (
            corners_start + self.lookdir[None, :] * 2*tr.linalg.norm(self.pos)
        )

        cone_lines = tr.stack((corners_start, corners_end), dim=1)
        plane_start_lines = tr.stack((corners_start, corners_start.roll(-1, dims=0)), dim=1)
        plane_end_lines = tr.stack((corners_end, corners_end.roll(-1, dims=0)), dim=1)

        segments = tr.concat((cone_lines, plane_start_lines, plane_end_lines))
        return [[segments, tr.ones(len(segments)), ['black'] * len(segments)]]