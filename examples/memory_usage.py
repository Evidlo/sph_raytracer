#!/usr/bin/env python3
# Peak memory requirement estimation

import numpy as np

# detector size
npix1 = 512
npix2 = 512

# volume size
nrad = 32
nele = 16
nazi = 32

# number of observations
nobs = 25
nchan = 2 # (e.g. multiple detectors in a spacecraft)

density = (1, nrad, nazi, nele)
density_size = 8 * np.prod(density) / 1e9 # dynamic density

# max number of voxels intersecting a ray
nvox_ray = 2 * (nrad + 1) + 2 * (nele + 1) + (nazi + 1)
# indices of intersecting voxels for each ray
indices = (nchan, nobs, npix1, npix2, nvox_ray, 3)
indices_dtype = 8 # integer dtype
# intersection lengths of intersecting voxels for each ray
lens = (nchan, nobs, npix1, npix2, nvox_ray)
lens_dtype = 8 # float dtype
# volume density values along each ray
densities = (nchan, nobs, npix1, npix2, nvox_ray)
densities_dtype = 8 # float dtype
# density values after sorting
aftersort = densities
aftersort_dtype = densities_dtype # float dtype

# total peak memory usage in GB
static_size = (
    (indices_dtype * np.prod(indices)) +
    (lens_dtype * np.prod(lens)) +
    (densities_dtype * np.prod(densities)) +
    (aftersort_dtype * np.prod(aftersort))
) / 1e9

print('\n--- Parameters ---\n')
print(f'({nrad}, {nele}, {nazi}) volume')
print(f'{nobs} observations, {nchan} channels, ({npix1}, {npix2}) sensor')
print('\n--- Memory Usage ---\n')
print('Ray coordinates memory:', static_size, 'GB')
print('Density memory:', density_size, 'GB')