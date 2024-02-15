# sph_raytracer

This is a volume raytracer implemented in PyTorch over a spherical coordinate system.  It was implemented very simply using only PyTorch array operations at the expense of memory consumption.

## Quickstart

    pip install sph_raytracer
    
    python example.py

![example output](example.png)

## Memory Usage

The peak memory usage in GB of this library can be approximated with the following expression

    #                        points             indices   intersection_length
    #                           |                    |       |
    mem = num_rays * (2 * nrad + 2 * nele + nazi) * (4 * 8 + 8) / 1e9
    
- `num_rays` - total number of rays across all measurement positions
- `nrad` - radial bins of volume
- `nele` - elevation bins of volume
- `nazi` - azimuthal bins of volume

## Running Tests

    pytest sph_raytracer/tests.py
    
## See Also

[tomosipo](https://github.com/ahendriksen/tomosipo), which inspired parts of this library's interface.
