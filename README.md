[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

# PyMagBot

## Description
PyMagBot is a Python module for modeling and control of magnetic microrobots. It is designed for versatility, modularity, and GPU acceleration using [PyTorch](https://pytorch.org/).

## Features
- Rigid transformations (SE(3))
- Frame hierarchy and kinematics
- GPU acceleration with [PyTorch](https://pytorch.org/)
- Integration with scientific libraries (NumPy, SciPy, Plotly)
- Future compatibility with ROS2

## Usage

See example in scripts folder


## Appendix

### Structure

```
pymagbot/ 
  ├── pymagbot/
              ├── __init__.py
  ├── pymagbot/math/
  │                ├── __init__.py
  │                ├── frame.py  # Frame, RigidTransform, RotationMatrix, Point, Direction
  │                └── geometry.py  # Geometric objects (to be implemented)
  ├── data/
  ├── docs/
  │   └── index.md  # Documentation index
  ├── notebook/
  ├── scripts/
  │   ├── example_frame.py  # Example usage of Frame
  │   └── example_geometry.py  # Example usage of Geometry (future)
  ├── tests/
  │   ├── test_frame.py  # Unit tests for Frame
  │   └── test_geometry.py  # Unit tests for Geometry
  ├── mkdocs.yml  # MkDocs configuration
  ├── pyproject.toml  # Modern package configuration
  ├── README.md  # Project overview
  ├── LICENSE  # License (MIT or CC BY 4.0)
  ├── CHANGELOG.md  # Project changes
  └── MANIFEST.in  # Packaging metadata
```
