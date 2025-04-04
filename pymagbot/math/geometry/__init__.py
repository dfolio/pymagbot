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
"""[pymagbot.math.geometry][] - geometry submodule

This subpackage provides classes and utilities for geometric objects that are defined
with respect to a given frame. It includes coordinate systems, meshes, vectors, and
utilities for geometric transformations.

"""

from pymagbot.math.geometry.coordinate import Coordinate
from pymagbot.math.geometry.geometry_base import GeometryBase
from pymagbot.math.geometry.mesh import Mesh
from pymagbot.math.geometry.span import Span
from pymagbot.math.geometry.vector import EuclideanVector
from pymagbot.math.geometry.vector_field import VectorField

# from pymagbot.math.geometry.operations import *
__all__ = [
    "GeometryBase",
    "Coordinate",
    "Span",
    "Mesh",
    "EuclideanVector",
    "VectorField",
]
