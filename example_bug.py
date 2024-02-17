#!/usr/bin/env python3

import torch as t
import matplotlib.pyplot as plt
import numpy as np
from glide.science.plotting import *

from sph_raytracer import SphericalVol, ConeRectGeom, Operator, ViewGeomCollection


vol = SphericalVol(shape=(1, 1, 4))

# theta = 2 * np.pi * (2/5)
# theta = np.pi
# pos = np.array((np.cos(theta), np.sin(theta), 0.5))
geom = ConeRectGeom(
    shape=(1, 1),
    pos=[-100, 0.00, 0.001],
    lookdir=[1, 0, 0],
    # fov=(10, 10)
    fov=(5, 5)
)
# geom.rays[:, :, :] = 0
# geom.rays[:, :, 0] = 1

op = Operator(vol, geom, debug=True, invalid=False)


# checkerboard
x = t.zeros(vol.shape)
# x[:, 25:, :] = 1
x[:, :, :] = 1

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
