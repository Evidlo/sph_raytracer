#!/usr/bin/env python3

import torch as t
import matplotlib.pyplot as plt
import numpy as np
from glide.science.plotting import *

from sph_raytracer import SphericalVol, ConeRectGeom, Operator, ViewGeomCollection

vol = SphericalVol(shape=(50, 50, 50), size=((3, 25), (0, np.pi), (-np.pi, np.pi)))
geoms = []
for theta in np.linspace(0, 2 * np.pi, 30):
    pos = np.array((np.cos(theta), np.sin(theta), .32))
    geoms.append(ConeRectGeom(
        shape=(150, 150),
        pos=200 * pos,
        lookdir=-pos / np.linalg.norm(pos),
        fov=(35, 35)
    ))

geom = ViewGeomCollection(*geoms)
op = Operator(vol, geom)

# test density with two nested shells
x = t.zeros(vol.shape)
x[-1, :, :] += 1
x[-10, :, :] += 1

# checkerboard
x = t.zeros(vol.shape)
s = 50
o = t.zeros((s, s))
o[:s//2, :s//2] = 1
o[s//2:, s//2:] = 1
o = t.tile(o, (vol.shape[1] // s + 1, vol.shape[2] // s + 1))[:vol.shape[1], :vol.shape[2]]
x[-1, :, :] = o
# x[-1, 25:, :] = 1
# x[-1:, :, :15] = o[:, :15]
# x[:, 25:29, :] = 1
# x[:, :, :] = 1

result = op(x)

save_gif('/srv/www/out.gif', result)

# plt.subplot(1, 2, 1)
# plt.title('Nested Shells')
# plt.imshow(result)
# ax = plt.subplot(1, 2, 2, projection='3d')
# op.plot(ax)
# ax.set_title('View Geometry')

# plt.savefig('example.png')
# plt.show()
