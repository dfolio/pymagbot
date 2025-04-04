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

This package contains modules for mathematical operations, spatial transformations,
and geometry objects used throughout PyMagBot.

Submodules
----------
- matrix_base: Defines the [`MatrixBase`][pymagbot.math.matrix_base.MatrixBase] class and related functionality.
- spatial: Spatial transformations and primitives ([`RotationMatrix`][pymagbot.math.spatial.RotationMatrix],
  [`RigidTransform`][pymagbot.math.spatial.RigidTransform],
  [`Vertex`][pymagbot.math.spatial.Vertex], [`Direction`][pymagbot.math.spatial.Direction], and
  [`Frame`][pymagbot.math.spatial.frame]).
- geometry: Geometric objects ([`Coordinate`][pymagbot.math.geometry.Coordinate],
  [`Span`][pymagbot.math.geometry.Span], [`Mesh`][pymagbot.math.geometry.Mesh],
  [`EuclideanVector`][pymagbot.math.geometry.EuclideanVector], [`VectorField`][pymagbot.math.geometry.VectorField],
  and coordinate system conversion utilities).

Tip
---
In `pymagbot`, **both 2D and 3D** transformation and primitives are supported.

"""

from pymagbot.math.matrix_base import MatrixBase
from pymagbot.math.spatial import (
    transform,
    primitives,
    frame,
)
from pymagbot.math.geometry import (
    geometry_base,
    coordinate,
    span,
    mesh,
    vector,
    vector_field,
)

__all__ = [
    "MatrixBase",
    # Spatial submodules
    "transform",
    "primitives",
    "frame",
    # Geometry submodules
    "geometry_base",
    "coordinate",
    "span",
    "mesh",
    "vector",
    "vector_field",
]