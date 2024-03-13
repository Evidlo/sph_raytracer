#!/usr/bin/env python3

import torch as t
import matplotlib.pyplot as plt

from sph_raytracer import SphericalGrid, ConeRectGeom, Operator
from sph_raytracer.plotting import image_stack

# define volume grid and viewing geometry vantage
vol = SphericalGrid(shape=(50, 50, 50))
# define a simple circular orbit around the origin
geoms = []
for theta in t.linspace(0, 2*t.pi, 10):
    geoms.append(ConeRectGeom(
        shape=(200, 200),
        pos=(5 * t.cos(theta), 5 * t.sin(theta), 1),
        fov=(45, 45)
    ))

# merge geometries together
geom = sum(geoms)

# define forward operator
# to run on CPU, use device='cpu'
op = Operator(vol, geom, device='cuda')

# test density with two nested shells
x = t.zeros(vol.shape, device=op.device)
x[:, 25:, :25] = 1
x[:, :25, 25:] = 1


result = op(x)

# ----- Plotting -----
# %% plot

plt.close('all')
fig = plt.figure(figsize=(10, 4))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2, projection='3d')

ax1.set_title('Nested Shells')
ani1 = image_stack(result, ax1, colorbar=True)

ax2.set_title('View Geometry')
ani2 = op.plot(ax2)

ani2.event_source = ani1.event_source
f = 'static_retrieval.gif'
print(f"Saving to {f}")
ani1.save(f, extra_anim=[ani2])

# plt.show()
