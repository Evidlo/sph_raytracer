#!/usr/bin/env python3

from sph_raytracer import *
from sph_raytracer.plotting import image_stack
import torch as t
import matplotlib.pyplot as plt

"""


              Z↑

              0
          ****|****
        ******|******
       * 0 ***|*** 0 *
       *******|*******                         .....o
      1-------+-------1  X→              ......
              |                   .......
         1    |    1
              |
              |
              2


        * : density=1
        o : ray_start
        . : ray

"""

grid = SphericalGrid(shape=(1, 2, 1), size_r=(3, 25))
geom = ConeRectGeom((20, 1), (200, 0, 1e-12), fov=(3.6, 3.6))
geom = ViewGeom(
    # doesn't work
    ray_starts=t.tensor([[200, 1e-12, 4e-12]]),
    rays=t.tensor([[-1, 3.1e-5, -3.1e-3]]),
    # works
    # ray_starts=t.tensor([[200, 1e-12, 4e-12]]),
    # rays=t.tensor([[-1, 3.1e-5, -3.2e-3]]),
)
geom.fov = (3.6, 3.6)



op = Operator(grid, geom, debug=True)

x = t.zeros(grid.shape)
x[:, 0, :] = 1

y = op(x)

print('y:', y)

# image_stack(y, geom, colorbar=True)
# plt.savefig('/tmp/out.png')


"""
>>> %run newbug.py
ray_start: tensor([2.0000e+02, 1.1682e-12, 3.9948e-12], dtype=torch.float64)
ray: tensor([-1.0000e+00,  3.0998e-05, -3.1610e-03], dtype=torch.float64)
typ   reg       intlen     dist      ind  neg
---------------------------------------------
?  r:[-1, 0, 0] l:0.00   t:0.00      i:-1 n:-1
r  r:[ 0, 0, 0] l:22.06  t:175.01    i:1  n:1
r  r:[-1, 0, 0] l:0.00   t:197.07    i:0  n:1
r  r:[ 0, 0, 0] l:22.06  t:202.93    i:0  n:0
r  r:[-1, 0, 0] l:0.00   t:224.99    i:1  n:0
e  r:[-1, 0, 0] l:0.00   t:inf       i:0  n:0
e  r:[-1, 1, 0] l:0.00   t:inf       i:1  n:0 <--- dist should be 0 instead of inf
e  r:[-1,-1, 0] l:0.00   t:inf       i:2  n:0
e  r:[-1, 0, 0] l:0.00   t:inf       i:0  n:0
e  r:[-1, 1, 0] l:0.00   t:inf       i:1  n:0 <--- dist should be 0 instead of inf
e  r:[-1,-1, 0] l:0.00   t:inf       i:2  n:0
a  r:[-1,-1, 0] l:0.00   t:inf       i:0  n:1
a  r:[-1,-1, 0] l:0.00   t:inf       i:1  n:1
y: tensor(44.1188, dtype=torch.float64)
"""