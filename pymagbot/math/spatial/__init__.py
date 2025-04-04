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
"""[spatial][pymagbot.math.spatial] - spatial module

This subpackage contains classes and functions for spatial transformations and
primitives used in PyMagBot. It provides operations on rotations, rigid transforms,
and a hierarchical frame system, as well as basic spatial primitives.
"""

from pymagbot.math.spatial.transform import RigidTransform, RotationMatrix
from pymagbot.math.spatial.primitives import Direction, Vertex
from pymagbot.math.spatial.operations import quaternion_to_rotation, euler_to_rotation, vector_to_rotation


__all__ = [
    "RotationMatrix", "RigidTransform",
    "Direction", "Vertex",
    "vector_to_rotation", "quaternion_to_rotation", "euler_to_rotation"
]
