# sph_raytracer

A 3D/4D volume raytracer in spherical coordinates for arbitrary detector shape implemented in PyTorch.

Check `examples/` for samples demonstrating forward raytracing and retrieval.

## Features

- 3D spherical raytracing with optional support for dynamic volume (4D)
- implemented purely in PyTorch for easy integration with PyTorch's optimization and machine learning capabilities
- support for square/circular detectors or other custom detector shapes
- optional retrieval framework for easily defining loss functions and parametric models (currently supports only static 3D volumes)

## Quickstart

    pip install -e .
    python examples/single_vantage.py

<img src="example.png" height=250/>

    python examples/static_retrieval.py

<p>
<img src="static_retrieval2.gif" height=200/>
<img src="static_retrieval1.gif" height=200/>
</p>

    python examples/dynamic_measurements.py
    
<img src="dynamic.gif" height=250/>

## Memory Usage

This library was uses only PyTorch array operations for implementation simplicity and speed at the expense of memory consumption.  The peak memory usage in GB can be approximated with `examples/memory_usage.py`

``` python
>>> %run examples/memory_usage.py

--- Parameters ---

(50, 50, 50) volume
50 observations, 1 channels, (50, 100) sensor

--- Memory Usage ---

Ray coordinates memory: 4.25 GB
Volume memory: 0.05 GB
```

## Architecture

Below is a list of modules in this package and their purpose:

Forward Raytracing

- `raytracer.py` - computation of voxel indices for intersecting rays, raytracing Operator
- `geometry.py` - viewing geometry (detector) definitions, spherical grid definition

Retrieval

- `model.py` - parameterized models for representing a density.  used in retrieval
- `loss.py` - some loss functions to be used in retrieval
- `retrieval.py` - retrieval algorithms
- `plotting.py` - functions for plotting stacks of images, retrieval losses

## Running Tests

    pytest sph_raytracer
    
## See Also

[tomosipo](https://github.com/ahendriksen/tomosipo), which inspired parts of this library's interface.
