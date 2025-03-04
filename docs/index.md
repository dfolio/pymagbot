[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

# PyMagBot

PyMagBot is a Python package for modeling and control of magnetic microrobots.

## Features
- Geometric operations (`Point`, `Direction`, `RotationMatrix`)
- Rigid body transformations (`SE(3)`)
- Hierarchical frame system (`Frame`)
- GPU acceleration using [PyTorch](https://pytorch.org/)
- Future ROS2 compatibility

## Quick Example

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

