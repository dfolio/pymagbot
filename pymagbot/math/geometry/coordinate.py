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
""" The `coordinate` module.
"""

from pymagbot.math.spatial.frame import Frame
from pymagbot.math.geometry.geometry_base import ComponentsBase
from pymagbot.math.spatial.primitives import Vertex


class Coordinate(ComponentsBase):
    """Represents a coordinate in a specific frame.

    A *coordinate* is a system that uses numbers to uniquely determine the position of a point in a specific
    [`frame`][pymagbot.math.frame.Frame].
    This is a thin wrapper around a [`Vertex`][pymagbot.math.spatial.primitives.Vertex], providing the
    [`GeometryBase`][pymagbot.math.geometry.GeometryBase]
    interface.

    Note
    -----
    In `pymagbot`, the coordinate system is mainly related to 2D (x-y) or 3D (x-y-z) Cartesian coordinate
    system. But, any projection can be used to switch to another coordinate system.

    Examples
    --------

    ```python
    from pymagbot.math.geometry import Coordinate

    # A default point centered at the origin
    p0 = Coordinate()

    # A point defined from its components
    p1 = Coordinate([1, 1, 1])
    ```

    Attributes
    ----------
    _components : Vertex
        The coordinate components.
    dim: int
        The base dimension (2 or 3) of the geometric object.
    frame : Frame
        The reference frame of the geometry object. Default the [world frame][pymagbot.math.frame.Frame.world] is
        used.
    name : str
        The unique name of the geometry object.
    """
    
    @property
    def cartesian(self) -> tuple:
        """The Cartesian coordinates."""
        return self.x, self.y, self.z
    
    @cartesian.setter
    def cartesian(self, c):
        """Set the Cartesian coordinates."""
        if len(c) != self.dim:
            raise ValueError(f"Expected {self.dim} components, got {len(c)}")
        self.components = c
    
    # TODO: Add properties for data coordinate systems (spherical, cylindrical, etc.)
    
    def express_in_frame(self, frame: Frame):
        """Express this `Coordinate` in a different `frame`.

        This method does not change the underlying geometry but returns a *new*
        geometry object whose properties (e.g., coordinates, components) are
        expressed with respect to the `target_frame`.

        Parameters
        ----------
        frame : Frame
            The frame in which to express the geometry.
        Returns
        -------
        Coordinate
            A *new* `Coordinate` object with its properties expressed in the `target_frame`.
        """
        new_coordinate = frame.local_rotation.data @ self._components.data + frame.local_translation.data
        return Coordinate(new_coordinate, dim=self.dim, frame=frame, name=f"{self.name}_in_{frame.name}")
