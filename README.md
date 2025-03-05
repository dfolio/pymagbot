[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![docs](https://shields.io/badge/MkDocs-documentation-informational)](https://mkdocstrings.github.io/python/)

# PyMagBot
## Description
PyMagBot is a Python module for modeling and control of magnetic microrobots. 
It is designed for versatility, modularity, and GPU acceleration using [PyTorch](https://pytorch.org/).

## Features
- Geometric operations (`Point`, `Direction`, `RotationMatrix`)
- Rigid body transformations (`SE(3)`)
- Hierarchical frame system (`Frame`)
- GPU acceleration using [PyTorch](https://pytorch.org/)
- Integration with scientific libraries (NumPy, SciPy, Plotly)
- Future ROS2 compatibility


## Usage


See the [Usage](usage) section of the docs for more examples!
Look also  to `scripts` folder.


### Quick Example

```python
from pymagbot.math.frame import Frame

world = Frame(name="world")
robot = Frame(name="robot", parent=world)

print(f"Robot's parent: {robot.parent.name}")
```

## Installation

```bash
git clone https://github.com/dfolio/pymagbot.git
cd pymagbot
pip install -e .
```
## Running the Documentation

To build and serve the documentation locally, run:

```bash
mkdocs serve
```
Then open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

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
