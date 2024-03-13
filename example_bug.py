#!/usr/bin/env python3

import torch as t
import matplotlib.pyplot as plt
import numpy as np
from glide.science.plotting import *

from sph_raytracer import SphericalGrid, ConeRectGeom, Operator, ViewGeomCollection


vol = SphericalGrid(shape=(50, 50, 50))
vol = SphericalGrid(shape=(1, 4, 1))

# theta = 2 * np.pi * (2/5)
# theta = np.pi
# pos = np.array((np.cos(theta), np.sin(theta), 0.5))
geom = ConeRectGeom(
    shape=(100, 100),
    pos=[5, 0, .02],
    lookdir=[-1, 0, 0],
    # fov=(3, 3)
    fov=(30, 30)
    # fov=(2, 2)
)
# geom.rays = geom.rays[5:6, 6:7]


op = Operator(vol, geom, debug=True, invalid=False)


# checkerboard
x = t.zeros(vol.shape)
# x[:, 25:, :] = 1
# x[:, 1, :] = 1
# x[:, 2, :] = 1
# x[:, 3, :] = 1
x[-1, vol.shape[1]//2:, :] = 1

result = op(x)
print(result)

plt.close()
plt.imshow(result)
plt.colorbar()
plt.savefig('/srv/www/out.png')

# plt.subplot(1, 2, 1)
# plt.title('Nested Shells')
# plt.imshow(result)
# ax = plt.subplot(1, 2, 2, projection='3d')
# op.plot(ax)
# ax.set_title('View Geometry')

# plt.savefig('example.png')
# plt.show()
