#!/usr/bin/env python3
# Evan Widloski - 2024-04-10
# Demonstration of tomographic retrieval of a a static volume

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
grid = SphericalGrid(shape=(50, 50, 50))

# generate a simple static test volume with two nested shells
x = t.zeros(grid.shape, device=op.device)
x[:, 25:, :25] = 1
x[:, :25, 25:] = 1

# define a simple circular orbit around the origin
geoms = []
for theta in t.linspace(0, 2*t.pi, 50):
    # use a circular detector
    geoms.append(ConeCircGeom(
        shape=(100, 50),
        pos=(5 * t.cos(theta), 5 * t.sin(theta), 1),
        fov=45
    ))

# merge view geometries together by adding
geoms = sum(geoms)

# define forward operator
# to run on CPU, use device='cpu'
op = Operator(grid, geoms, device='cuda')

# generate some measurements to retrieve from.  No measurement noise in this case
meas = op(x)

# ----- Retrieval -----
# choose a parametric model for retrieval.
# FullyDenseModel is the most basic and simply has 1 parameter per voxel
# see model.py for how to define your own parametric models
m = FullyDenseModel(grid)

# choose loss functions and regularizers with weights
# see loss.py for how to define your own loss/regularization
loss_fns = [1 * SquareLoss(), 1 * NegRegularizer()]

retrieved = gd(op, meas, m, loss_fns=loss_fns, num_iterations=100)

# ----- Plotting -----
# %% plot
matplotlib.use('Agg')
plt.close('all')

print('plotting...')
fig1 = plt.figure(figsize=(8, 4))
fig2 = plt.figure(figsize=(8, 4))
ax1 = fig1.add_subplot(1, 2, 1)
ax2 = fig1.add_subplot(1, 2, 2)
ax3 = fig2.add_subplot(1, 2, 1, polar=True)
ax4 = fig2.add_subplot(1, 2, 2, projection='3d')

# generate rotating 3D preview of ground truth
ax1.set_title('Truth')
ani1 = image_stack(preview3d(x, grid), ax=ax1, colorbar=True)

# generate rotating 3D preview of retrieval
ax2.set_title('Retrieved')
ani2 = image_stack(preview3d(retrieved[0], grid), ax=ax2, colorbar=True)

ani2.event_source = ani1.event_source
ani1.save('static_retrieval1.gif', extra_anim=[ani2])

# plot stack of measurements in an animation
ax3.set_title('Measurements')
ani3 = image_stack(meas, geoms, ax=ax3, colorbar=True)

# generate a 3D wireframe animation of viewing geometries
ax4.set_title('View Geometry')
ani4 = op.plot(ax=ax4)
ani4.save('static_retrieval2.gif', fps=30, extra_anim=[ani3])

# plt.show()
