# Notes on `Generators_3D.py`

Companion explanation file for the synthetic image generators in the `logic/` package.

This module creates synthetic binary 3D images that mimic porous media. The generated images are
used as test inputs for the filters, metrics, and simulations in the rest of the project.

## Available generators

- `BLOBS`: creates a blob-based random porous structure.
- `CYLINDERS`: generates a porous medium made from random cylinders.
- `OVERLAPPING_SPHERES`: creates overlapping spherical inclusions/void regions.
- `POLYDISPERSE_SPHERES`: creates a sphere packing with a distribution of sizes.
- `PSEUDO_ELECTROSTATIC_PACKING`: generates a pseudo-random packing of spheres using a simple
  electrostatic-inspired placement algorithm.
- `PSEUDO_GRAVITY_PACKING`: creates a packing influenced by a gravity-like direction.
- `VORONOI_EDGES`: generates a network-like structure based on Voronoi edges.

## Notes

- All generators use PoreSpy's back-end routines and save a `.tif` file in the working directory.
- The functions are deterministic by default because the RNG is seeded before generation.
- The `invert` argument can switch the binary phase convention (void vs. solid) without changing
  the geometry in a fundamental way.

## Typical usage

Use these generators to test the UI or to build reference images for benchmarking new metrics and
simulations before applying them to real imported data.
