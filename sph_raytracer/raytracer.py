#!/usr/bin/env python3

from collections import namedtuple
import math
import torch as tr

from .geometry import ViewGeomCollection

# shorthand for creating new axes
na = None

DEVICE = 'cpu'
FTYPE = tr.float64
ITYPE = tr.int8

@tr.jit.script
def forward_fill_jit(x, initial, dim=-1, fill_what=0, inplace=False):
    """Forward fill arbitrary dimension Pytorch tensor over specific axis

    Args:
        x (tensor): tensor with values to forward fill
        initial (tensor or None): initial fill value.  If `x.shape` is (1, 2, 3, 4)
            and dim==-2, then `initial.shape` should be (1, 2, 4)
        dim (int): dimension to fill
        fill_what (float): value to be replaced
        inplace (bool): whether to make a copy of `t`

    Returns:
        t (tensor): tensor with filled in values
    """
    if not inplace:
        x = x.clone()

    # move fill dim to the front to keep indexing simple
    x = x.moveaxis(dim, 0)
    last = initial

    for i in range(x.shape[0]):
        x[i] = x[i].where(x[i]!=fill_what, last)
        last = x[i]

    # move fill dim back to original location
    return x.moveaxis(0, dim)


def trace_indices(vol, xs, rays, ftype=FTYPE, itype=ITYPE, device=DEVICE, invalid=False, debug=False):
    """Sort points by distance.  Then filter out invalid intersections (nan t values)
    and points which lie outside radius `max_r` (inplace)

    Args:
        vol (SphericalVol): spherical grid
        xs (tensor): starting points of rays (*num_rays, 3)
        rays (tensor): directions of rays (*num_rays, 3)
        ftype (torch dtype): type specification for floats
        itype (torch dtype): type specification for ints
        invalid (bool): filter out invalid lengths/regions

    Returns:
        inds (tensor[int]): voxel indices of every voxel that ray intersects with
            (*rays.shape, max_int_voxels, 3)
        lens (tensor[float]): intersection length of each voxel with ray's path
            (*rays.shape, max_int_voxels)

    where `max_int_voxels` is `2*vol.shape[0] + 2*vol.shape[1] + vol.shape[2]`

    """
    spec = {'dtype': ftype, 'device': device}
    ispec = {'dtype': itype, 'device': device}

    print('--------- 1')
    # --- compute voxel indices for all rays and their distances ---
    print('--------- 1.1')
    r_t, _r_regs, _, _r_inds, _r_ns = r_torch(vol.rs, xs, rays, ftype=ftype, itype=itype, device=device)
    print('--------- 1.2')
    e_t, _e_regs, _, _e_inds, _e_ns = e_torch(vol.phis, xs, rays, ftype=ftype, itype=itype, device=device)
    print('--------- 1.3')
    a_t, _a_regs, _, _a_inds, _a_ns = a_torch(vol.thetas, xs, rays, ftype=ftype, itype=itype, device=device)

    print('--------- 1.4')
    # concatenate intersection distances/points from all geometry kinds
    all_ts = tr.cat((r_t, e_t, a_t), dim=-1)
    # del r_t, e_t, a_t
    # concatenate regions and place into appropriate column
    # FIXME: cleaner dtype/device handling?
    # FIXME: using -2 to represent invalid region index
    r_regs = tr.full((*_r_regs.shape, 3), -2, device=device, dtype=itype)
    r_regs[..., 0] = _r_regs
    e_regs = tr.full((*_e_regs.shape, 3), -2, device=device, dtype=itype)
    e_regs[..., 1] = _e_regs
    a_regs = tr.full((*_a_regs.shape, 3), -2, device=device, dtype=itype)
    a_regs[..., 2] = _a_regs
    all_regs = tr.cat((r_regs, e_regs, a_regs), dim=-2)
    _all_regs = tr.cat((_r_regs, _e_regs, _a_regs), dim=-1)
    print('--------- 2')

    # del r_regs, e_regs, a_regs, _r_regs, _e_regs, _a_regs

    # mark regions behind ray start as invalid
    # all_regs[all_ts < 0] = -2
    # _all_regs[all_ts < 0] = -2

    # sort points by distance
    # https://discuss.pytorch.org/t/sorting-and-rearranging-multi-dimensional-tensors/148340
    all_ts_s, s = all_ts.sort(dim=-1)
    _all_regs_s = _all_regs.gather(-1, s)
    # s_expanded = s[..., None].repeat_interleave(3, dim=-1)
    # all_regs_s = all_regs.gather(1, s_expanded)
    all_regs_s = tr.take_along_dim(all_regs, s[..., None], dim=-2)
    print('--------- 3')

    forward_fill_jit(
        all_regs_s,
        # tr.full_like(all_regs_s, -2)[..., 0, :],
        # find_starts(vol, rays),
        find_starts(vol, xs),
        dim=-2, fill_what=-2, inplace=True
    )
    print('--------- 4')

    # segment intersection lengths with voxels
    # last segment in each ray is infinitely long
    inf = tr.full(all_ts_s.shape[:-1] + (1,), float('inf'), **spec)
    all_lens_s = all_ts_s.diff(dim=-1, append=inf)

    print('--------- 5')

    if not invalid:
        # zero out nan/inf lengths
        invalid = all_lens_s.isinf() + all_lens_s.isnan()
        all_lens_s[invalid] = 0

        # set invalid regions to 0 and zero associated segment length
        all_lens_s[all_regs_s[..., 0] > vol.shape[0] - 1] = 0
        all_lens_s[all_regs_s[..., 1] > vol.shape[1] - 1] = 0
        all_lens_s[all_regs_s[..., 2] > vol.shape[2] - 1] = 0
        # all_regs_s[all_regs_s[..., 0] > vol.shape[0] - 1] = 0
        # all_regs_s[all_regs_s[..., 1] > vol.shape[1] - 1] = 0
        # all_regs_s[all_regs_s[..., 2] > vol.shape[2] - 1] = 0

        all_lens_s[all_regs_s[..., 0] < 0] = 0
        all_lens_s[all_regs_s[..., 1] < 0] = 0
        all_lens_s[all_regs_s[..., 2] < 0] = 0
        # all_regs_s[all_regs_s[..., 0] < 0] = 0
        # all_regs_s[all_regs_s[..., 1] < 0] = 0
        # all_regs_s[all_regs_s[..., 2] < 0] = 0
    print('--------- 6')

    if debug:
        r_inds = tr.full((*_r_inds.shape, 3), -2, device=device, dtype=itype)
        r_inds[..., 0] = _r_inds
        e_inds = tr.full((*_e_inds.shape, 3), -2, device=device, dtype=itype)
        e_inds[..., 1] = _e_inds
        a_inds = tr.full((*_a_inds.shape, 3), -2, device=device, dtype=itype)
        a_inds[..., 2] = _a_inds
        all_inds = tr.cat((r_inds, e_inds, a_inds), dim=-2)
        _all_inds = tr.cat((_r_inds, _e_inds, _a_inds), dim=-1)
        _all_inds_s = _all_inds.gather(-1, s)
        all_inds_s = tr.take_along_dim(all_inds, s[..., None], dim=-2)
        _all_ns = tr.cat((_r_ns, _e_ns, _a_ns), dim=-1)
        _all_ns_s = _all_ns.gather(-1, s)
        _all_kinds = tr.cat((tr.full_like(_r_inds, 0), tr.full_like(_e_inds, 1), tr.full_like(_a_inds, 2)), dim=-1)
        _all_kinds_s = _all_kinds.gather(-1, s)

        shp = len(all_regs_s.shape)
        if shp == 4:
            which = (0, 0)
        elif shp == 3:
            which = (0,)
        else:
            raise ValueError("Wrong shape {all_regs_s.shape}")
        regs = all_regs_s[which]
        lens = all_lens_s[which]
        ts   = all_ts_s[which]
        inds = _all_inds_s[which]
        ns = _all_ns_s[which]
        kinds = _all_kinds_s[which]
        kmap = {0:'r', 1:'e', 2:'a'}
        print(find_starts(vol, xs))
        for k, r, l, t_, ind, n in zip(kinds, regs, lens, ts, inds, ns):
            print(
                f'{kmap[int(k)]:<2}',
                f'r:[{r[0]:>2},{r[1]:>2},{r[2]:>2}]',
                f'l:{float(l):<4.2f}',
                f't:{float(t_):<10.2f}'
                f'i:{int(ind):<2}',
                f'n:{n:<2}',
                # f'p:[{p[0]:>4.1f},{p[1]:>4.1f},{p[2]:>4.1f}]',
            )

    # FIXME: pytorch requires int64 for indexing
    # r, e, a = all_regs_s.moveaxis(-1, 0).type(tr.int64)
    # return (r, e, a), all_lens_s
    return tuple(all_regs_s.moveaxis(-1, 0).type(tr.int64)), all_lens_s


def isclose(a, b, factor=4):
    """Detect whether a/b are close.  Like tr.isclose but scales with dtype

    Args:
        a (tensor): input tensor
        b (tensor): input tensor
        factor (float): allow larger errors

    Returns:
        tensor
    """
    # detecting whether value is very small to avoid precision issues
    # `resolution` is a bit more forgiving than `eps` (also tr.isclose doesn't scale with dtype)
    return abs(a - b) < tr.finfo(a.dtype).resolution ** (1/factor)

def r_torch(rs, xs, rays, ftype=FTYPE, itype=ITYPE, device=DEVICE):
    """Compute intersections of ray with concentric spheres

    Args:
        rs (tensor): radius of each sphere
        xs (tensor): starting points of rays (*num_rays, 3)
        rays (tensor): directions of rays (*num_rays, 3)
        ftype (torch dtype): type specification for floats
        itype (torch dtype): type specification for ints
        device (str): torch device

    Returns:
        t (tensor): distance of each point from x along ray
            (*num_rays, 2 * num_spheres).  Can be negative
        regions (tensor[int]): region index associated with each point
            (*num_rays, num_spheres).
        points (tensor): intersection points of rays with spheres
            (*num_rays, 2 * num_spheres, 3)
        inds (tensor[int]): geometry index that the point lies on
            (*num_rays, 2 * num_spheres)
        negative_crossing (tensor[int]): whether ray crosses geometry in
            positive or negative direction (*num_rays, 2 * num_spheres)

    Ref: https://kylehalladay.com/blog/tutorial/math/2013/12/24/Ray-Sphere-Intersection.html
    """
    spec = {'dtype': ftype, 'device': device}
    ispec = {'dtype': itype, 'device': device}
    assert len(rs) - 1 < tr.iinfo(ispec['dtype']).max, "Too many rs!  Would cause overflow"

    xs = tr.asarray(xs, **spec)
    rays = tr.asarray(rays, **spec)
    rs = tr.asarray(rs, **spec)
    rshape = rays.shape[:-1] # number and shape of rays
    na_rays = (na,) * len(rshape) # for creating single dimensions for ray shape

    rays /= tr.linalg.norm(rays, axis=-1)[..., na] # (*num_rays, 3)

    dotproduct = lambda a, b: tr.einsum('...j,...j->...', a, b)

    tc = dotproduct(-xs, rays) # (*num_rays)
    d = tr.sqrt(dotproduct(xs, xs) - tc**2) # (*num_rays)
    # NOTE: run out of memory when doing below for 512x512, 50obs
    # t1c = tr.sqrt(rs[na, :]**2 - d[:, na]**2) # (*num_rays, num_spheres)
    # NOTE: this is the same as above but uses less memory
    t1c = tr.empty((*rshape, len(rs)), **spec)
    t1c[...] = rs[na_rays + (Ellipsis,)]**2 # (*num_rays, num_spheres)
    t1c[...] -= d[..., na]**2 # (*num_rays, num_spheres)
    t1c = tr.sqrt(t1c)

    t = tr.empty((*rshape, 2 * len(rs)), **spec)
    t[..., :len(rs)], t[..., len(rs):] = tc[..., na] - t1c, tc[..., na] + t1c
    inds = tr.cat((tr.arange(len(rs), **ispec), tr.arange(len(rs), **ispec)))
    inds = inds.repeat(*rshape, 1)
    del tc, t1c

    # NOTE: run out of memory when doing below for 512x512, 50obs
    # points = rays[..., na, :] * t[..., na] + xs[..., na, :]
    # NOTE: this is the same as above but uses less memory
    points = tr.empty((*rshape, 2 * len(rs), 3), **spec)
    points[...] = rays[..., na, :]
    points[...] *= t[..., na]
    points[...] += xs[..., na, :]

    # compute region index
    # check whether crossing of plane is positive or negative
    dotproduct = lambda a, b: tr.einsum('...c,...bc->...b', a, b)
    negative_crossing = (dotproduct(rays, points) < 0).type(tr.int8)

    regions = inds - negative_crossing

    # mark region outside outermost shell as invalid
    regions[regions == len(rs) - 1] = -1

    return t, regions, points, inds, negative_crossing


def e_torch(phis, xs, rays, ftype=FTYPE, itype=ITYPE, device=DEVICE):
    """Compute intersections of rays with elevation cones

    Args:
        phis (tensor): Number of elevation cones
            or cone elevations (radians)
        xs (tensor): starting points of rays (*num_rays, 3)
        rays (tensor): directions of rays (*num_rays, 3)
        ftype (torch dtype): type specification for floats
        itype (torch dtype): type specification for ints
        device (str): torch device

    Returns:
        t (tensor): distance of each point from x along ray
            (*num_rays, 2 * num_cones).  Can be negative
        regions (tensor[int]): region index associated with each point
            (*num_rays, 2 * num_cones).
        points (tensor): intersection points of rays with cones
            (*num_rays, 2 * num_cones, 3)
        inds (tensor[int]): geometry index that the point lies on
            (*num_rays, 2 * num_cones)
        negative_crossing (tensor[int]): whether ray crosses geometry in
            positive or negative direction (*num_rays, 2 * num_cones)

    Ref: http://lousodrome.net/blog/light/2017/01/03/intersection-of-a-ray-and-a-cone/
    Ref: "Intersection of a Line and a Cone", David Eberly, Geometric Tools
    """
    spec = {'dtype': ftype, 'device': device}
    ispec = {'dtype': itype, 'device': device}

    assert len(phis) - 1 < tr.iinfo(ispec['dtype']).max, "Too many phis!  Would cause overflow"

    print('--------- 1.2.1')
    zero = tr.tensor(0, **spec)
    xs = tr.asarray(xs, **spec)
    rays = tr.asarray(rays, **spec)
    phis = tr.asarray(phis, **spec)
    rshape = rays.shape[:-1] # number and shape of rays
    na_rays = (na,) * len(rshape) # for creating single dimensions for ray shape

    rays /= tr.linalg.norm(rays, axis=-1)[..., None] # (*num_rays, 3)

    # (*num_rays, num_cones)

    v = tr.tensor((0, 0, 1), **spec)

    print('--------- 1.2.2')

    dotproduct = lambda a, b: tr.einsum('...j,...j->...', a, b)
    a = rays[..., 2:]**2 - (tr.cos(phis)**2)[na_rays + (Ellipsis,)]
    b = 2 * (rays[..., 2:] * xs[..., 2:] - dotproduct(rays, xs)[..., None] * (tr.cos(phis)**2)[na_rays + (Ellipsis,)])
    c = xs[..., 2:]**2 - (tr.linalg.norm(xs, axis=-1)**2)[..., None] * (tr.cos(phis)**2)[na_rays + (Ellipsis,)]
    a[isclose(a, zero)] = zero

    # a = dotproduct(rays, v)[:, None] - (tr.cos(phis)**2)[None, :]
    # b = 2 * (dotproduct(rays, v) *)

    print('--------- 1.2.3')

    # ray not parallel to cone
    delta = b**2 - 4*a*c
    delta[isclose(delta, zero)] = zero

    t1 = (-b + tr.sqrt(delta)) / (2 * a)
    t2 = (-b - tr.sqrt(delta)) / (2 * a)

    # --- ray intersecting cone ---
    # compute single or double intersection
    is_single = isclose(delta, zero)
    is_single = tr.logical_and(isclose(a, zero), tr.logical_not(isclose(b, zero)))
    t_normal = tr.empty((*rshape, 2 * len(phis)), **spec)
    t_normal[..., :len(phis)] = tr.where(is_single, -2*c / b, t1)
    t_normal[..., len(phis):] = tr.where(is_single, float('inf'), t2)
    del t1, t2
    print('--------- 1.2.4')

    # --- ray parallel to cone ---
    t_parallel = tr.empty((*rshape, 2 * len(phis)), **spec)
    t_parallel[..., :len(phis)] = -c / b
    t_parallel[..., len(phis):] = float('inf')

    is_parallel = tr.full_like(t_normal, False, device=spec['device'], dtype=tr.bool)
    is_parallel[..., :len(phis)] = tr.logical_and(isclose(a, zero), tr.logical_not(isclose(b, zero)))
    is_parallel[..., len(phis):] = is_parallel[..., :len(phis)]
    t = tr.where(is_parallel, t_parallel, t_normal)
    # del t_normal, t_parallel, is_parallel
    print('--------- 1.2.5')

    # --- ray lies on cone ---
    t[..., :len(phis)][(a==0) * (b==0) * (c==0)] = float('inf')
    t[..., len(phis):][(a==0) * (b==0) * (c==0)] = float('inf')
    # t[..., :len(phis)][(a==0) * (b==0) * (c==0)] = 0
    # t[..., len(phis):][(a==0) * (b==0) * (c==0)] = 0

    inds = tr.cat((tr.arange(len(phis), **ispec), tr.arange(len(phis), **ispec)))
    inds = inds.repeat(*rshape, 1)
    print('--------- 1.2.6')

    points = rays[..., na, :] * t[..., :, na] + xs[..., na, :]

    # compute region index
    # compute a normal plane at intersection point
    points_normal = tr.cross(
        points,
        tr.stack(
            (
                -points[..., 1],
                points[..., 0],
                tr.zeros_like(points[..., 0]),
            ),
            axis=-1
        )
    )
    print('--------- 1.2.7')
    # points_normal /= tr.linalg.norm(points_normal, dim=-1)[..., na]
    print('--------- 1.2.7.1')
    # check whether crossing of plane is positive or negative
    dotproduct = lambda a, b: tr.einsum('...c,...bc->...b', a, b)
    prod = dotproduct(rays, points_normal)
    print('--------- 1.2.7.2')
    negative_crossing = (prod > 0).type(tr.int8)
    print('--------- 1.2.7.3')
    regions = inds - negative_crossing

    # ray just barely glances a cone, keep the region the same
    # FIXME: this error factor is hard to get right in advance.  need to do
    # a proper forward analysis of floating-point forward error propagation
    # to find upper bound on error at this point
    # https://www-users.cselabs.umn.edu/classes/Fall-2019/csci5304/FILES/LecN4.pdf
    print('--------- 1.2.7.4')
    regions[isclose(prod, zero, factor=5)] = -2
    print('--------- 1.2.8')

    # filter out intersections with opposite shadow cone
    phis_expanded = phis.repeat(2)
    # cone_point_z = tr.cos(phis_expanded) * tr.linalg.norm(points, axis=-1)
    # shadow = tr.logical_not(isclose(points[..., 2], cone_point_z))
    cone_point_z_sign = tr.cos(phis_expanded) >= 0
    shadow = tr.logical_not((points[..., 2] >= 0) == cone_point_z_sign)
    # when phi==pi/2, sign is unreliable.  Coincidentally, shadow masking is not necessary
    # for this case
    shadow[..., isclose(tr.tensor(tr.pi / 2, **spec), phis_expanded)] = False


    points[shadow] = float('inf')
    t[shadow] = float('inf')

    # mark region outside last cone as invalid
    regions[regions == len(phis) - 1] = -1

    return t, regions, points, inds, negative_crossing


def a_torch(thetas, xs, rays, ftype=FTYPE, itype=ITYPE, device=DEVICE):
    """Compute intersections of rays with azimuth planes

    Args:
        thetas (tensor): plane angles (radians)
        xs (tuple): starting points of rays (num_rays, 3)
        rays (tuple): directions of rays (num_rays, 3)
        ftype (torch dtype): type specification for floats
        itype (torch dtype): type specification for ints
        device (str): torch device

    Returns:
        t (tensor): distance of each point from x along ray
            (*num_rays, num_planes).  Can be negative
        regions (tensor[int]): region index associated with each point
            (*num_rays, num_planes).
        points (tensor[float]): intersection points of rays with planes
            (*num_rays, num_planes, 3)
        inds (tensor[int]): geometry index that the point lies on
            (*num_rays, num_planes)
        negative_crossing (tensor[int]): whether ray crosses geometry in
            positive or negative direction (*num_rays, num_planes)

    """
    spec = {'dtype': ftype, 'device': device}
    ispec = {'dtype': itype, 'device': device}

    assert len(thetas) - 1 < tr.iinfo(ispec['dtype']).max, "Too many thetas!  Would cause overflow"

    zero = tr.tensor(0, **spec)
    xs = tr.asarray(xs, **spec)
    rays = tr.asarray(rays, **spec)
    thetas = tr.asarray(thetas, **spec)
    rshape = rays.shape[:-1] # number and shape of rays
    na_rays = (na,) * len(rshape) # for creating single dimensions for ray shape

    planes = tr.stack((tr.cos(thetas), tr.sin(thetas), tr.zeros_like(thetas, **spec)), dim=-1)
    plane_norms = tr.stack((-tr.sin(thetas), tr.cos(thetas), tr.zeros_like(thetas, **spec)), dim=-1)

    dotproduct = lambda a, b: tr.einsum('...bc,...jc->...b', a, b)
    # distance along ray
    t = (
        -dotproduct(plane_norms[na_rays + (Ellipsis, Ellipsis)], xs[..., na, :]) /
        dotproduct(plane_norms[na_rays + (Ellipsis, Ellipsis)], rays[..., na, :])
    )
    inds = tr.arange(len(thetas), **ispec)
    inds = inds.repeat(*rshape, 1)
    # compute region index - check whether Z component of cross product is negative

    cross = tr.cross(planes[na_rays + (Ellipsis, Ellipsis)], rays[..., na, :])[..., -1]
    # ray is parallel to plane
    # FIXME: wrap up into nice isclose func
    is_parallel = tr.isclose(cross, zero, atol=tr.finfo(cross.dtype).resolution)
    t[..., is_parallel] = float('inf')

    negative_crossing = (cross < 0).type(tr.int8)
    regions = inds - negative_crossing

    # if thetas are full range, wrap around
    if -thetas[0] == thetas[-1] == tr.pi:
        regions = regions % (len(thetas) - 1)
    else:
        # mark region outside last plane as invalid,
        regions[regions == len(thetas) - 1] = -1

    # FIXME: can't handle case when ray goes directly through Z axis!

    # NOTE: run out of memory when doing below for rshape (50, 512, 512)
    # points = xs[..., na, :] + t[..., :, na] * rays[..., na, :]
    # this is the same as above but uses less memory
    points = tr.empty((*rshape, len(thetas), 3), **spec)
    points[...] = t[..., :, na]
    points[...] *= rays[..., na, :]
    points[...] += xs[..., na, :]

    shadow = tr.einsum('bc,...bc->...b', planes[..., :2], points[..., :2]) < 0

    points[shadow] = float('inf')
    t[shadow] = float('inf')

    return t, regions, points, inds, negative_crossing


def cart2sph(xyz):
    """Convert cartesian coordinates to spherical coordinates
    https://stackoverflow.com/a/72609701/7465444

    Args:
        xyz (tuple): cartesian coordinates (x, y, z)

    Returns:
        spherical (tuple): spherical coordinates (radius, elevation, azimuth),
            where elevation is measured from Z-axis in radians [0, ℼ] and
            azimuth is measured from X-axis in radians [-ℼ, ℼ]
    """
    x, y, z = xyz.moveaxis(-1, 0)

    rea = tr.empty_like(xyz, dtype=float)

    pre_selector = ((slice(None),) * rea.ndim)[:-1]

    xy_sq = x ** 2 + y ** 2
    rea[(*pre_selector, 0)] = tr.sqrt(xy_sq + z ** 2)
    rea[(*pre_selector, 1)] = tr.arctan2(tr.sqrt(xy_sq), z)
    rea[(*pre_selector, 2)] = tr.arctan2(y, x)

    return rea


def sph2cart(rea):
    """Convert spherical coordinates to cartesian coordinates

    Args:
        spherical (tuple): spherical coordinates (radius, elevation, azimuth),
            where elevation is measured from Z-axis in radians [0, ℼ] and
            azimuth is measured from X-axis in radians [-ℼ, ℼ]

    Returns:
        cartesian (tuple): cartesian coordinates (x, y, z)
    """
    r, e, a = rea.moveaxis(-1, 0)

    xyz = tr.empty_like(rea)

    pre_selector = ((slice(None),) * xyz.ndim)[:-1]

    xyz[(*pre_selector, 0)] = r * tr.sin(e) * tr.cos(a)
    xyz[(*pre_selector, 1)] = r * tr.sin(e) * tr.sin(a)
    xyz[(*pre_selector, 2)] = r * tr.cos(e)

    return xyz


def find_starts(vol, xs, ftype=FTYPE, device=DEVICE):
    """Compute voxel indices of ray start location at infinity

    Args:
        rays (tensor): directions of rays (*num_rays, 3)

        vol (SphericalVol): spherical grid
        spec (dict): type specification for floats
        ftype (torch dtype): type specification for floats
        device (str): torch device

    Returns:
        regions (tensor)
    """
    spec = {'dtype': ftype, 'device': device}

    rs, phis, thetas = (vol.rs, vol.phis, vol.thetas)
    rays, rs, phis, thetas = map(lambda x: tr.asarray(x, **spec), (xs, rs, phis, thetas))
    rays_sph = cart2sph(rays)
    # starting radius of rays is infinite
    # rays_sph[..., 0] = float('inf')

    # make contiguous to avoid pytorch searchsorted warnings
    rays_r = rays_sph[..., 0].contiguous()
    rays_e = rays_sph[..., 1].contiguous()
    rays_a = rays_sph[..., 2].contiguous()

    # find region where each ray starts
    r_reg = tr.searchsorted(rs, rays_r, right=True) - 1
    e_reg = tr.searchsorted(phis, rays_e, right=True) - 1
    a_reg = tr.searchsorted(thetas, rays_a, right=True) - 1

    # consider rays lying on top of last geometry as valid and set appropriate index
    r_reg = tr.where(rays_r == rs[-1], vol.shape[0] - 1, r_reg)
    e_reg = tr.where(rays_e == phis[-1], vol.shape[1] - 1, e_reg)
    a_reg = tr.where(rays_a == thetas[-1], vol.shape[2] - 1, a_reg)

    # if ray starts in an invalid region, set the region index to -1
    r_reg[r_reg == vol.shape[0]] = -1
    e_reg[e_reg == vol.shape[1]] = -1
    a_reg[a_reg == vol.shape[2]] = -1

    return tr.stack((r_reg, e_reg, a_reg), axis=-1)


class Operator:
    """Raytracing operator

    Args:
        vol (SphericalVol): spherical grid extent/resolution information
        geom (ViewGeom): measurement locations and rays
        dynamic (bool): whether input density is evolving (4D) or static (3D)
        ftype (torch dtype): type specification for floats
        itype (torch dtype): type specification for ints
        device (str): torch device
    """
    def __init__(self, vol, geom, dynamic=False,
                 ftype=FTYPE, itype=ITYPE, device=DEVICE,
                 debug=False, invalid=False):
        self.vol = vol
        self.geom = geom
        self.dynamic = dynamic
        self.ftype = ftype
        self.itype = itype
        self.regs, self.lens = trace_indices(
            vol, geom.ray_starts, geom.rays,
            ftype=ftype, itype=itype, device=device,
            invalid=invalid, debug=debug
        )

        if dynamic and not isinstance(geom, ViewGeomCollection):
            raise ValueError("geom must be ViewGeomCollection instance when dynamic=True")

    def __call__(self, density):
        """Lookup up density indices for all rays and compute
        inner-product with intersection length

        Args:
            density (tensor): 3D tensor of shape `vol.shape` if dynamic=False.  4D tensor
                with first dimension equal to length of geom.shape[0] if dynamic=True

        Returns:
            line_integrations (tensor): integrated lines of sight of shape `geom.shape`
        """
        # FIXME: does branching here affect torch.compile?
        if self.dynamic:
            t = tr.arange(len(self.geom))[:, None, None, None]
            return (density[(t, *self.regs)] * self.lens).sum(axis=-1)
        else:
            return (density[self.regs] * self.lens).sum(axis=-1)

    def __repr__(self):
        if self.dynamic:
            return f"Operator({(self.geom.shape[0], *self.vol.shape)} → {self.geom.shape})"
        else:
            return f"Operator({self.vol.shape} → {self.geom.shape})"


    def plot(self, fig=None, ax=None):
        """Generate Matplotlib wireframe plot for this object

        Returns
            matplotlib Animation
        """
        import matplotlib.pyplot as plt
        from matplotlib import animation
        from mpl_toolkits.mplot3d.art3d import Line3DCollection

        if fig is None and ax is None:
            fig = plt.figure(figsize=(3, 3))
            ax = fig.add_subplot(projection='3d', computed_zorder=False)

        self.vol.plot(ax)

        wireframe = self.geom._wireframe

        # segments, widths, colors = wireframe[0]
        # lc = Line3DCollection(segments, linewidths=widths, colors=colors)
        lc = Line3DCollection([])

        def update(num):
            segments, widths, colors = wireframe[num]
            lc.set_segments(segments)
            lc.set_linewidth(widths)
            lc.set_colors(colors)
            return lc,

        # limits and labels
        # lim = max(tr.linalg.norm(self.geom.ray_starts, dim=-1))
        lim = tr.abs(self.geom.ray_starts).max()
        ax.set_xlim3d([-lim, lim])
        ax.set_xlabel('X')
        ax.set_ylim3d([-lim, lim])
        ax.set_ylabel('Y')
        ax.set_zlim3d([-lim, lim])
        ax.set_zlabel('Z')

        ax.add_collection(lc)

        # some stupid matplotlib stuff
        # from matplotlib import rcParams
        # from matplotlib.tight_bbox import adjust_bbox
        # renderer = fig.canvas.get_renderer()
        # bbox_inches = fig.get_tightbbox(renderer, )
        # pad_inches = rcParams['savefig.pad_inches']
        # bbox_inches = bbox_inches.padded(pad_inches)
        # adjust_bbox(fig, bbox_inches, fig.canvas.fixed_dpi)

        # fix whitespace
        fig.subplots_adjust(left=0, top=1, bottom=0.1, right=.95, wspace=0, hspace=0)

        N = len(wireframe)
        # ani = animation.FuncAnimation(fig, update, N, interval=3000/N, blit=True)
        ani = animation.FuncAnimation(fig, update, N, blit=True)
        # def save(*args, **kwargs):
        #     kwargs.setdefault()
        #     return ani.save(*args, **kwargs)

        # ani.save = save
        return ani