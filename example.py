#!/usr/bin/env python3

import torch as t
import matplotlib.pyplot as plt

from sph_raytracer import SphericalVol, ConeRectGeom, Operator

vol = SphericalVol(shape=(50, 50, 50))
geom = ConeRectGeom(
    shape=(256, 256),
    pos=(5, 0, 0),
    lookdir=(-1, 0, 0),
    fov=(45, 45)
)

op = Operator(vol, geom)

# test density with two nested shells
x = t.zeros(vol.shape)
x[-1, :, :] += 1
x[-10, :, :] += 1

result = op(x)

plt.title('Nested Shells')
plt.subplot(1, 2, 1)
plt.imshow(result)
ax = plt.subplot(1, 2, 2, projection='3d')
op.plot(ax)
plt.show()
