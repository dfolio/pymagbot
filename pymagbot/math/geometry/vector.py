from typing import Union

import torch
from numpy.typing import ArrayLike

from pymagbot.math.spatial.frame import Frame
from pymagbot.math.geometry.geometry_base import ComponentsBase
from pymagbot.math.matrix_base import MatrixBase
from pymagbot.math.spatial.primitives import Direction, Vertex


class EuclideanVector(ComponentsBase):
    """Represents a Euclidean vector with magnitude and direction.

    A Euclidean vector, or simply a _vector_, is a geometric object that has both **magnitude** and **direction**. It is
    represented by a point in space and a direction.
    This is a thin wrapper around a [`Vertex`][pymagbot.math.spatial.primitives.Vertex], providing
    the [`GeometryBase`][pymagbot.math.geometry.GeometryBase] interface.

    Attributes:
    -----------
    components : Vertex
        The components of the vector.
    dim: int
        The base dimension (2 or 3) of the geometric object.
    frame : Frame
        The reference frame of the geometry object. Default the [world frame][pymagbot.math.frame.Frame.world] is
        used.
    name : str
        The unique name of the geometry object.
    magnitude : float
        The magnitude of the vector.
    direction : Direction
        The direction of the vector.
    """
    
    @property
    def magnitude(self) -> torch.Tensor:
        """Get the magnitude  (i.e. Euclidean norm) of the vector."""
        return torch.linalg.vector_norm(self.components.data, ord=2)
    
    @magnitude.setter
    def magnitude(self, mag: float):
        """The vector magnitude (Euclidean norm)"""
        self.scale(mag / self.magnitude)
    
    @property
    def direction(self) -> Direction:
        """Get the direction of the vector."""
        return Direction(self.components.data)
    
    @direction.setter
    def direction(self, direction: Union[ArrayLike, torch.Tensor, list, tuple, MatrixBase]):
        """Set the direction of the vector."""
        self.components = Vertex(direction) * self.magnitude
    
    def express_in_frame(self, frame: Frame) -> 'EuclideanVector':
        """Express this EuclideanVector in a different `frame`.

        This method does not change the underlying geometry but returns a *new*
        geometry object whose properties (e.g., coordinates, components) are
        expressed with respect to the `target_frame`.

        Parameters
        ----------
        frame : Frame
            The frame in which to express the geometry.
        Returns
        -------
        EuclideanVector
            A *new* EuclideanVector object with its properties expressed in the `target_frame`.
        """
        vector = frame.local_rotation.data @ self.components.data
        return EuclideanVector(vector, dim=self.dim, frame=frame, name=f"{self.name}_in_{frame.name}")
