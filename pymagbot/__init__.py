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
pymagbot - Python Module for Magnetic Microrobot Simulation and Control.

This package provides tools for modeling and controlling magnetic microrobots.
It includes submodules for handling  spatial transformations, reference frames, and geometric objects.

Submodules
----------
- `math`: Mathematical operations, spatial transformations, frames and geometric primitives.
- `utils`: soon...
"""

# from pymagbot import math
from ._config import __author__, DEVICE, DTYPE, FORMAT, USE_CUDA
from ._version import __version__

__all__ = ['math', 'USE_CUDA', 'DEVICE', 'DTYPE', "FORMAT"]


# __all__.extend(math.__all__)

def _lazy_import():
    """Lazy import to prevent circular dependencies."""
    global MatrixBase, RotationMatrix, RigidTransform, Frame, get_frame
    global GeometryBase, Coordinate, Span, Mesh, EuclideanVector, VectorField
    global Direction, Vertex
    global rotation_vector, quaternion_to_rotation, rotation_to_euler
    
    from pymagbot.math.matrix_base import MatrixBase
    from pymagbot.math.spatial import RotationMatrix, RigidTransform
    from pymagbot.math.frame import Frame, get_frame
    from pymagbot.math.geometry import GeometryBase, Coordinate, Span, Mesh, EuclideanVector, VectorField
    from pymagbot.math.spatial_primitives import Direction, Vertex
    from pymagbot.math.transform import vector_to_rotation, quaternion_to_rotation, euler_to_rotation


_lazy_import()
