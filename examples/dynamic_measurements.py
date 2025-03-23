#!/usr/bin/env python3
# Evan Widloski - 2024-04-10
# Demonstration of tomographic measurements of a dynamic volume

import torch as t
import matplotlib.pyplot as plt
import matplotlib

from sph_raytracer import SphericalGrid, ConeRectGeom, ConeCircGeom, Operator
from sph_raytracer.plotting import image_stack, preview3d
from sph_raytracer.model import FullyDenseModel
from sph_raytracer.retrieval import gd
from sph_raytracer.loss import SquareLoss, NegRegularizer

# ----- Setup -----

# define volume grid.  Grid spacing may be customized but are left default here
grid = SphericalGrid(shape=(20, 50, 50, 50))

# generate a simple static test volume with two nested shells
# to run on CPU, use device='cpu'
x = t.zeros(grid.shape, device='cuda')
x[:, :, 25:, :25] = 1
x[:, :, :25, 25:] = 1
# moving elevation element
for time in range(grid.shape.t):
    x[time, :, time*2, :] += 1

# define a simple circular orbit around the origin
geoms = []
for theta in t.linspace(0, 2*t.pi, grid.shape.t):
    geoms.append(
        # use a circular shaped detector (pointed at origin by default)
        ConeCircGeom(
            shape=(100, 50),
            pos=(5 * t.cos(theta), 5 * t.sin(theta), 1),
            fov=(0, 45)
        )
    )

# merge view geometries together by adding
geoms = sum(geoms)

# define forward operator
op = Operator(grid, geoms, device=x.device)

# generate some measurements to retrieve from.  No measurement noise in this case
meas = op(x)

# ----- Plotting -----
# %% plot

from sph_raytracer.plotting import preview3d, image_stack, color_negative
import matplotlib.pyplot as plt
import matplotlib
plt.close('all')
matplotlib.use('Agg')

print('plotting...')
fig = plt.figure(figsize=(12, 4))

ax1 = fig.add_subplot(1, 3, 1)
ax2 = fig.add_subplot(1, 3, 2, projection='3d')
ax3 = fig.add_subplot(1, 3, 3, polar=True)

# generate rotating 4D preview of ground truth
ani1 = image_stack(preview3d(x, grid, azim=-45, orbit=False), ax=ax1)
ax1.set_title("Dynamic Volume Preview")

# generate a 3D wireframe animation of viewing geometries
ax2.set_title('View Geometry')
ani2 = op.plot(ax=ax2)

# show actual raytracer measurements
ani3 = image_stack(meas, geom=geoms, ax=ax3)
# plt.title("Measurements")

ani3.event_source = ani2.event_source = ani1.event_source
ax3.set_title("Measurements")
ani1.save('dynamic.gif', fps=15, extra_anim=[ani2, ani3])