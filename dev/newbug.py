#!/usr/bin/env python3

from sph_raytracer import *
from sph_raytracer.plotting import image_stack
import torch as t
import matplotlib.pyplot as plt

grid = SphericalGrid(shape=(1, 2, 1), size_r=(3, 25))
# geom = ViewGeom(rays=rays, ray_starts=ray_starts)
# geom.fov = (3.6, 3.6)
geom = ConeRectGeom((20, 1), (200, 0, 1e-12), fov=(3.6, 3.6))
geom = ViewGeom(
    # ray_starts=t.tensor([[200, 0, 1e-12]]),
    ray_starts=t.tensor([[200, 0.1, 0.1]]),
    # rays=t.tensor([[-1, 0, .004]]) # works
    # rays=t.tensor([[-1, 0, 0]]) # works
    # rays=t.tensor([[-1, 0, .003]]) # broken
)


# op = Operator(grid, geom, debug=True, debug_los=(9, 0), invalid=False)
op = Operator(grid, geom, debug=True, invalid=False)

x = t.zeros(grid.shape)
x[:, 0, :] = 1

y = op(x)

print('y:', y)

# image_stack(y, geom, colorbar=True)
plt.savefig('/tmp/out.png')