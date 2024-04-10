#!/usr/bin/env python3

import torch as t
import matplotlib.pyplot as plt
import matplotlib

from sph_raytracer import SphericalGrid, ConeRectGeom, ConeCircGeom, Operator
from sph_raytracer.plotting import image_stack, preview3d
from sph_raytracer.model import FullyDenseModel
from sph_raytracer.retrieval import gd
from sph_raytracer.loss import SquareLoss, NegRegularizer

# define volume grid and viewing geometry vantage
grid = SphericalGrid(shape=(50, 50, 50))
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

# test density with two nested shells
x = t.zeros(grid.shape, device=op.device)
x[:, 25:, :25] = 1
x[:, :25, 25:] = 1

meas = op(x)

# ----- Retrieval -----
# choose a model for retrieval
m = FullyDenseModel(grid)
# choose loss functions and regularizers with weights
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

ax1.set_title('Truth')
ani1 = image_stack(preview3d(x, grid), ax=ax1, colorbar=True)

ax2.set_title('Retrieved')
ani2 = image_stack(preview3d(retrieved[0], grid), ax=ax2, colorbar=True)

ani2.event_source = ani1.event_source
ani1.save('static_retrieval1.gif', extra_anim=[ani2])

ax3.set_title('Measurements')
ani3 = image_stack(meas, geoms, ax=ax3, colorbar=True)

ax4.set_title('View Geometry')
ani4 = op.plot(ax=ax4)
ani4.save('static_retrieval2.gif', fps=30, extra_anim=[ani3])


# plt.show()
