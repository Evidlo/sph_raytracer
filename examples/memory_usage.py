#!/usr/bin/env python3
# Peak memory requirement estimation

import numpy as np

# detector size
npix1 = 50
npix2 = 100

# volume size
nrad = 50
nele = 50
nazi = 50

# number of observations
nobs = 50
nchan = 1 # (e.g. multiple detectors in a spacecraft)

density = (nobs, nrad, nazi, nele)
density_size = 8 * np.prod(density) / 1e9 # dynamic density

# max number of voxels intersecting a ray
nvox_ray = 2 * nrad + 2 * nele + nazi
# indices of intersecting voxels for each ray
indices = (nchan, nobs, npix1, npix2, nvox_ray, 3)
indices_dtype = 8
# intersection lengths of intersecting voxels for each ray
lens = (nchan, nobs, npix1, npix2, nvox_ray)
lens_dtype = 2
# volume density values along each ray
densities = (nobs, npix1, npix2, nvox_ray)
densities_dtype = 2

# total peak memory usage in GB
static_size = 2 * (
    (indices_dtype * np.prod(indices)) +
    (lens_dtype * np.prod(lens)) +
    (densities_dtype * np.prod(densities))
) / 1e9

print('\n--- Parameters ---\n')
print(f'({nrad}, {nele}, {nazi}) volume')
print(f'{nobs} observations, {nchan} channels, ({npix1}, {npix2}) sensor')
print('\n--- Memory Usage ---\n')
print('Ray coordinates memory:', static_size, 'GB')
print('Density memory:', density_size, 'GB')