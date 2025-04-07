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

In `pymagbot` a *geometry object* is a mathematical concept, equivalent to a choice of
language used to describe observations in a space. For instance, an observer in an
observational _frame of reference_ can choose to employ any coordinate system
(Cartesian, cylindrical, spherical, generalized...) to describe observations
made from this frame. A change in the choice of this coordinate system does not
change the observer's motion state, and therefore does not result in a change in
the observer's observational reference frame.

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
