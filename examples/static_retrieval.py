#!/usr/bin/env python3

import torch as t
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')

from sph_raytracer import SphericalGrid, ConeRectGeom, Operator
from sph_raytracer.plotting import image_stack, preview3d
from sph_raytracer.model import FullyDenseModel
from sph_raytracer.retrieval import gd
from sph_raytracer.loss import SquareLoss

# define volume grid and viewing geometry vantage
grid = SphericalGrid(shape=(50, 50, 50))
# define a simple circular orbit around the origin
geoms = []
for theta in t.linspace(0, 2*t.pi, 10):
    geoms.append(ConeRectGeom(
        shape=(200, 200),
        pos=(5 * t.cos(theta), 5 * t.sin(theta), 1),
        fov=(45, 45)
    ))

# merge view geometries together by adding
geom = sum(geoms)

# define forward operator
# to run on CPU, use device='cpu'
op = Operator(grid, geom, device='cuda')

# test density with two nested shells
x = t.zeros(grid.shape, device=op.device)
x[:, 25:, :25] = 1
x[:, :25, 25:] = 1

meas = op(x)

# ----- Retrieval -----
# choose a model for retrieval
m = FullyDenseModel(grid)
# choose loss functions and regularizers with weights
loss_fns = [1 * SquareLoss()]
retrieved = gd(op, meas, m, loss_fns=loss_fns, num_iterations=50)

# ----- Plotting -----
# %% plot

plt.close('all')
fig = plt.figure(figsize=(10, 4))
ax1 = fig.add_subplot(1, 3, 1)
ax2 = fig.add_subplot(1, 3, 2)
ax3 = fig.add_subplot(1, 3, 3, projection='3d')

ax1.set_title('Truth')
ani1 = image_stack(preview3d(x), ax1, colorbar=True)

ax2.set_title('Retrieved')
ani2 = image_stack(preview3d(retrieved[0]), ax2, colorbar=True)
ani2.event_source = ani1.event_source

ax3.set_title('View Geometry')
ani3 = op.plot(ax3)
ani3.event_source = ani1.event_source

f = 'static_retrieval.gif'
print(f"Saving to {f}")
ani1.save(f, extra_anim=[ani2, ani3])

# plt.show()
