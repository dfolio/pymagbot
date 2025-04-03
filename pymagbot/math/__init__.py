#  pymagbot
#  Copyright (c) 2025 dfolio
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
[pymagbot.math][] - Mathematical and geometric utilities for the `pymagbot` package.

This module provides spatial transformations for **rigid body motion** in SE(2) and SE(3).
It includes classes for rotation matrices and rigid transformations.

Features
--------
- `MatrixBase`: Abstract base class for tensor operations.
- `RotationMatrix`: SO(2) and SO(3) rotation handling.
- `RigidTransform`: SE(2) and SE(3) transformations.
- `Frame`: Reference frames for hierarchical transformations.

Example
-------
```python
from pymagbot.math.spatial import RotationMatrix
_rotation = RotationMatrix.from_euler([0.1, 0.2, 0.3])
print(_rotation.to_quaternion())
```

Submodules
----------
- `math.matrix_base`: Abstract base class for matrix-like objects.
- `math.spatial`: Rotation matrices and rigid transformations.
- `math.spatial_primitives`: Basic spatial primitives (Vertex, Direction).
- `math.frame`: Reference frames (coordinate systems).
- `math.geometry`: Frame-dependent geometric objects (Coordinates, Mesh, etc.).
- `math._transform`: Utility functions for transformation computations.

Tip
---
In `pymagbot` **both 2D and 3D transformations** are supported.

"""

from pymagbot._config import DTYPE, USE_CUDA
from pymagbot.math.frame import Frame, get_frame
from pymagbot.math.geometry import Coordinate, EuclideanVector, GeometryBase, Mesh, Span, VectorField
from pymagbot.math.matrix_base import MatrixBase
from pymagbot.math.spatial import RigidTransform, RotationMatrix
from pymagbot.math.spatial_primitives import Direction, Vertex
from pymagbot.math.transform import quaternion_to_rotation, euler_to_rotation, vector_to_rotation

__all__ = [
    "USE_CUDA", "DTYPE",
    "MatrixBase", "RotationMatrix", "RigidTransform",
    "Frame", "get_frame",
    "GeometryBase", "Coordinate", "Span", "Mesh", "EuclideanVector", "VectorField",
    "Direction", "Vertex",
    "vector_to_rotation", "quaternion_to_rotation", "euler_to_rotation"
]
