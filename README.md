[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![docs](https://img.shields.io/badge/MkDocs-documentation-informational)](https://mkdocstrings.github.io/python/)

# PyMagBot

## Description
PyMagBot is a Python module designed for the modeling and control of magnetic microrobots. It emphasizes versatility, modularity, and high-performance computations through GPU acceleration with [PyTorch](https://pytorch.org/). PyMagBot provides a comprehensive suite of mathematical and physical modeling tools including geometric operations, rigid body transformations, and a hierarchical frame system—all essential for simulating micro/nano-robotic systems.

## Features
- **GPU Acceleration:** Leverages [PyTorch](https://pytorch.org/) for fast and precise computations.
- **Mathematical Operations:**
  - Base operations via `MatrixBase` and `VectorBase`
  - Spatial transformations with `RotationMatrix` and `RigidTransform`
  - Geometric primitives such as `Vertex`, `Direction`, and coordinate conversion operations
- **Hierarchical Frame System:** Manage and manipulate reference frames using the `Frame` class.
- **Geometric Objects:** Support for `Coordinate`, `Span`, `Mesh`, `EuclideanVector`, and `VectorField`
- **Integration:** Designed to work seamlessly with scientific libraries (NumPy, SciPy, Plotly, Pandas) and planned integration with ROS2.
- **Modular and Extensible:** Clear separation of mathematical, physical, and device-level functionalities.


## Requirements and Dependencies
PyMagBot is built on top of several robust Python packages. The core dependencies are:

- **[PyTorch](https://pytorch.org/):** For tensor computations and GPU acceleration.
- **[NumPy](https://numpy.org/):** For numerical operations and array management.
- **[SciPy](https://scipy.org/):** (Optional) For additional scientific computing functionality.
- **[Plotly](https://plotly.com/):** (Optional) For interactive visualization of simulations.
- **[Pandas](https://pandas.pydata.org/):** (Optional) For handling tabular data (e.g., in workspace management).

Additional dependencies include:
- **[MkDocs](https://www.mkdocs.org) and [mkdocstrings](https://mkdocstrings.github.io/):** For generating documentation.
- **YAML:** For configuration parsing (via PyYAML).

## Documentation
### Installation

Clone the repository and install PyMagBot in editable mode:

```bash
git clone https://github.com/dfolio/pymagbot.git
cd pymagbot
pip install -e .
```
### Running the Documentation

The complete API documentation is generated with [MkDocs](https://www.mkdocs.org) and [mkdocstrings](https://mkdocstrings.github.io/).
To build and serve the documentation locally:

```bash
mkdocs serve
```
Then open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

### Usage


See the [Usage](usage/index.md) section of the docs for more examples!
Look also  to `scripts` folder.


### Quick Example

```python
from pymagbot.math.frame import Frame

world = Frame(name="world")
robot = Frame(name="robot", parent=world)

print(f"Robot's parent: {robot.parent.name}")
```

For coordinate conversions:

```python
from pymagbot.math.spatial_primitives import Vertex, VectorBase
from pymagbot.math.geometry.operations import vertex_cartesian_to_spherical

# Create a vertex with Cartesian coordinates
pos = Vertex([1.0, 1.0, 2.0])
r, theta, phi = vertex_cartesian_to_spherical(pos)
print(f"Spherical coordinates: r={r:.2f}, theta={theta:.2f}, phi={phi:.2f}")
```
## Contributing
Contributions are welcome! Please refer to the [CONTRIBUTING](CONTRIBUTING.md) guidelines for details on how to propose improvements or report issues.

## License
PyMagBot is licensed under the GPLv3. See attached [LICENSE](LICENSE) for details.

## Changelog
Refer to [CHANGELOG](CHANGELOG.md) for a history of changes.

## Future Developments

- Magnetic Field & Forces: Compute and manage magnetic fields using various approximations, and calculate magnetic 
forces acting on microrobots.
- Device Management:
  - Develop a parameter management system (ParamsBase) for variable parameters.
  - Implement modules for electromagnetic coils and microrobots with unique naming, reference frames, and design parameters.
- Workspace Modeling: Define simulation workspaces with customizable physical properties (density, viscosity, etc.).

- ROS2 Integration: Develop real-time control interfaces and integrate with ROS2.
- Advanced Visualization & Logging: Improve plotting and logging utilities to support complex simulations and debugging.


### Roadmap
PyMagBot is under active development. 
Refer to the [ROADMAP.md](ROADMAP.md) for detailed plans and upcoming features.

## Appendix

### References







