#!/usr/bin/env python3

import numpy as np

# Peak memory requirement estimatator

# detector
npix1 = 50
npix2 = 100

# volume size
nrad = 50
nele = 50
nazi = 50

# number of observations
nobs = 50
nchan = 1

density = (nobs, nrad, nazi, nele)
density_size = 8 * np.prod(density) / 1e9 # dynamic density

# max number of voxels intersecting a ray
nvox_ray = 2 * nrad + 2 * nele + nazi
indices = (nchan, nobs, npix1, npix2, nvox_ray)
lens = (nchan, nobs, npix1, npix2, nvox_ray)
densities = (nobs, npix1, npix2, nvox_ray)
static_size = 2 * ((3 * 8 * np.prod(indices)) + (8 * np.prod(lens)) + (2 * np.prod(densities))) / 1e9

print('\n--- Parameters ---\n')
print(f'({nrad}, {nele}, {nazi}) volume')
print(f'{nobs} observations, {nchan} channels, ({npix1}, {npix2}) sensor')
print('\n--- Memory Usage ---\n')
print('Ray coordinates memory:', static_size, 'GB')
print('Density memory:', density_size, 'GB')