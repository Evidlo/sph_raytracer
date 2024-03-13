#!/usr/bin/env python3

import torch as t
import matplotlib.pyplot as plt

from sph_raytracer import SphericalGrid, ConeRectGeom, Operator

# define volume grid and viewing geometry vantage
vol = SphericalGrid(shape=(50, 50, 50))
geom = ConeRectGeom(
    shape=(256, 256),
    pos=(5, 0, 0),
    fov=(45, 45)
)

# define forward operator
# to run on CPU, use device='cpu'
op = Operator(vol, geom, device='cuda')

# test density with two nested shells
x = t.zeros(vol.shape, device=op.device)
x[-1, :, :] += 1
x[-10, :, :] += 1

result = op(x)

# ----- Plotting -----

plt.close('all')
fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2, projection='3d')

ax1.set_title('Nested Shells')
ax1.imshow(result.detach().cpu())

ax2.set_title('View Geometry')
ani = op.plot(ax2)

plt.show()
