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
geometry - The `geometry` module help manage objects that are attached to a [`Frame`][pymagbot.math.frame.Frame].

In `pymagbot` any *geometry object* is a mathematical concept, equivalent to a choice of
language used to describe observations in a space. For instance, an observer in an
observational _frame of reference_ can choose to employ any coordinate system
(Cartesian, cylindrical, spherical, generalized...) to describe observations
made from this frame. A change in the choice of this coordinate system does not
change the observer's motion state, and therefore does not result in a change in
the observer's observational reference frame.
"""

import itertools
import weakref
from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
from numpy.typing import ArrayLike

from pymagbot._config import DEVICE, DTYPE, USE_CUDA
from pymagbot.math.frame import Frame, get_frame
from pymagbot.math.matrix_base import MatrixBase, parse_dict
from pymagbot.math.spatial import RigidTransform
from pymagbot.math.spatial_primitives import Direction, Vertex

__all__ = ['GeometryBase', 'Coordinate', 'Span', 'Mesh', 'EuclideanVector', 'VectorField']

known_geometry_objects: weakref.WeakValueDictionary[str, 'GeometryBase'] = weakref.WeakValueDictionary()
""" Dictionary to track known geometry objects (ensuring uniqueness)"""


class GeometryBase(MatrixBase):
    """Abstract base class for geometry objects.

    Geometry objects are mathematical concepts used to describe observations in a space.
    They have a unique `name` and is associated with a `frame` of reference.
    Any change in the `frame` implies that the geometric object will follow it.
    
    Note
    ----
    Any operation (add, sub, mult, eq...) are assumed to be realized in the same frame.
        
    Warning
    -------
    The `GeometryBase` class inherits all the [`MatrixBase`][pymagbot.math.matrix_base.MatrixBase] operators.
    Be careful when using them.  Even if the operation does not raise an error and is successful, it does not mean
    that the result is correct.
    
    Examples
    --------
    ```python
    #TODO: add usage examples here
    ```
    
    Attributes
    ----------
    data : torch.Tensor
        The base tensor data.
    dim: int
        The base dimension (2 or 3) of the geometric object.
    frame : Frame
        The reference frame of the geometry object. Default the [world frame][pymagbot.math.frame.Frame.world] is
        used.
    name : str
        The unique name of the geometry object.
    """
    _id_counter = itertools.count(1)  # Unique ID generator for geometry objects
    _name = ''
    
    def __new__(cls, data: Optional[Union[torch.Tensor, ArrayLike, 'MatrixBase', list, tuple]] = None,
                dim: int = 3, frame: Optional[Union[Frame, dict, str]] = None,
                name: Optional[str] = None, *args, **kwargs) -> 'GeometryBase':
        """Create a new [`GeometryBase`][pymagbot.math.geometry.GeometryBase] instance or retrieve an existing one with
         the same name.
         """
        name: str = name.lower() if name else f"{cls.__name__}_{next(cls._id_counter)}"
        if name in known_geometry_objects:
            return known_geometry_objects[name]  # Return existing object if it already exists
        
        instance = super().__new__(cls)
        instance._name = name  # Store the name early
        known_geometry_objects[name] = instance  # Register the geometry object
        return instance
    
    def __init__(self, data: Optional[Union[torch.Tensor, ArrayLike, 'MatrixBase', list, tuple]] = None,
                 dim: int = None, frame: Optional[Union[Frame, dict, str]] = None, name: Optional[str] = None):
        """Initialize the GeometryBase.

        Parameters
        ----------
        data : array_like, optional
            The geometry object data.r
        dim : int, optional
            The  geometry object dimension
        frame : Frame, optional
            The unique frame of the geometry object. If `None`, the [`world`][pymagbot.math.frame.Frame.world] frame is
            used.
        name : str, optional
            The unique name of the geometry object. The name of a geometry object must be unique (case-insensitive).
            If `None`, a name is automatically generated.
        """
        if hasattr(self, 'initialized'):
            return  # Prevent double initialization
        super().__init__(data, dim=dim)
        self.name: str = self._name  # Use the name set in __new__
        self._frame = get_frame(frame) if frame is not None else Frame.world
        self.initialized: bool = True
    
    def __iter__(self):
        yield from {
            "data" : self.data,
            "frame": self.frame,
            "name" : self.name
        }.items()
    
    def __repr__(self) -> str:
        """String representation of the object."""
        return f"{self.__class__.__name__}(name='{self.name}')"
    
    def to_dict(self, full: bool = True) -> dict:
        """Converts the [`GeometryBase`][pymagbot.math.geometry.GeometryBase] object to a dictionary

        Parameters
        ----------
        full: bool, optional
            Whenever export all the structure or only the relevant parts.
            Default set to `True`.
        Returns
        -------
        dict
            The object structure as a dictionary.
        """
        dict_obj: dict = {"type" : self.__class__.__name__,
                          "frame": self.frame.name if isinstance(self.frame, Frame) else 'None',
                          "name" : self.name}
        if full:
            dict_obj['data'] = self.data.tolist()
        return dict_obj
    
    @classmethod
    def _from_dict(cls, dict_obj: Optional[Union[dict, str]], known: Optional[dict] = None):
        dict_obj = parse_dict(dict_obj)
        device = torch.device("cuda" if USE_CUDA else "cpu")
        _type = dict_obj.get('type', None)
        name = dict_obj.get('name', None)
        data = torch.as_tensor(dict_obj['data'], device=device, dtype=DTYPE) if 'data' in dict_obj else torch.empty(0)
        frame = get_frame(dict_obj['frame'], known) if 'frame' in dict_obj else Frame.world
        return _type, data, name, frame
    
    @classmethod
    def from_dict(cls, dict_obj: Optional[Union[dict, str]], known: Optional[dict] = None):
        """ Create a  [`GeometryBase`][pymagbot.math.geometry.GeometryBase] object from a dictionary

        Parameters
        ----------
        dict_obj: Union[dict, str]
            The imported dictionary.
        known: dict, optional
            A dictionary of known frames (see also [`get_frame`][pymagbot.math.frame.get_frame])
        Returns
        -------
        GeometryBase
            The  [`GeometryBase`][pymagbot.math.geometry.GeometryBase] object.
        Raises
        ------
        TypeError
            If the `dict_obj` is not a [`GeometryBase`][pymagbot.math.geometry.GeometryBase] type.
        
        TODO: use `_type` to check if `dict_obj` is the same as cls.
        """
        _, data, name, frame = cls._from_dict(dict_obj, known)
        return cls(data=data, frame=frame, name=name)
    
    def update(self):
        """Update attributes/properties when `frame` is changing"""
        pass
    
    @staticmethod
    def known() -> weakref.WeakValueDictionary:
        """Get the list of known geometry objects."""
        return known_geometry_objects
    
    @property
    def name(self) -> str:
        """Get the name of the geometric object."""
        return self._name
    
    @name.setter
    def name(self, name: str):
        """Set/Change the name of the geometric object."""
        name: str = name.lower()
        if name not in known_geometry_objects:
            known_geometry_objects.pop(self._name, None)  # Remove the old name
            self._name = name.lower()
            known_geometry_objects[self._name] = self  # Add the new name
    
    @property
    def frame(self) -> Frame:
        """Get the reference frame of the geometry object."""
        return self._frame
    
    @frame.setter
    def frame(self, frame: Union[Frame, dict, str]):
        """Set/Change the reference frame of the geometry object."""
        self.change_frame(frame)
    
    def change_frame(self, frame: Union[Frame, dict, str], known: Optional[dict] = None):
        """Change the local [`Frame`][pymagbot.math.frame.Frame] of the
        [`GeometryBase`][pymagbot.math.geometry.GeometryBase] object.
        """
        frame = get_frame(frame, known)
        if not isinstance(frame, Frame):
            raise TypeError("frame must be a Frame object")
        self._frame = frame
        self.update()
    
    # def move(self, location: Union[ArrayLike, torch.Tensor, list, tuple], local: bool = True):
    #     """Move the [`GeometryBase`][pymagbot.math.geometry.GeometryBase]
    #     [`frame`][pymagbot.math.geometry.GeometryBase.frame].
    #
    #     Parameters
    #     ----------
    #     location: array_Like
    #         The position vector of the `Geometry` object origin wrt the reference frame.
    #     local : bool, optional
    #         If the move is with respect to the local or global frame. Defaults to `True`.
    #     """
    #     self.frame.move(location=location, local=local)
    #     self.update()
    #
    # def rotate(self, rotation_matrix: Union[ArrayLike, torch.Tensor, list, tuple], local: bool = True):
    #     """Rotate the [`GeometryBase`][pymagbot.math.geometry.GeometryBase]
    #     [`frame`][pymagbot.math.geometry.GeometryBase.frame].
    #
    #     Parameters
    #     ----------
    #     rotation_matrix : RotationMatrix
    #         The rotation matrix.
    #     local : bool, optional
    #         Whether to rotate in the local frame
    #
    #     See Also
    #     --------
    #     Frame: Which implements the [`Frame.rotate`][pymagbot.math.frame.Frame.rotate] method.
    #     """
    #     self.frame.rotate(rotation=rotation_matrix, local=local)
    #     self.update()
    
    @abstractmethod
    def express_in_frame(self, frame: Frame) -> 'GeometryBase':
        """Express this geometry in a different `frame`.
        
        This method does not change the underlying geometry but returns a *new*
        geometry object whose properties (e.g., coordinates, vectors) are
        expressed with respect to the `target_frame`.

        Parameters
        ----------
        frame : Frame
            The frame in which to express the geometry.
        Returns
        -------
        GeometryBase
            A *new* GeometryBase object with its properties expressed in the `target_frame`.
        """
        pass
    
    def express_in_world(self) -> 'GeometryBase':
        """Express this geometry in the global `world` frame.

        This method does not change the underlying geometry but returns a *new*
        geometry object whose properties (e.g., coordinates, vectors) are
        expressed with respect to the `target_frame`.

        Returns
        -------
        GeometryBase
            A *new* GeometryBase object with its properties expressed in the `target_frame`.
        """
        return self.express_in_frame(Frame.world)


class ComponentsBase(GeometryBase, ABC):
    """Abstract base class for geometry objects with components.
    
    A components is a geometric primitive described with a 2 or 3 components (or vector).

    Attributes
    ----------
    components : Vertex
        The components of the Geometry Object.
    dim: int
        The base dimension (2 or 3) of the geometric object.
    frame : Frame
        The reference frame of the geometry object. Default the [world frame][pymagbot.math.frame.Frame.world] is
        used.
    name : str
        The unique name of the geometry object.
    """
    
    def __init__(self, components: Union[Vertex, ArrayLike, torch.Tensor, list, tuple],
                 dim: int = None, frame: Frame = None, name: Optional[str] = None):
        """Initialize a [`ComponentsBase`][pymagbot.geometry.ComponentsBase].

        Parameters
        ----------
        components : Union[Vertex, ArrayLike, torch.Tensor, list, tuple]
            The coordinates of the point.
        dim : int, optional
            The dimension of the coordinate (2 or 3).
        frame : Frame, optional
            The reference frame of the coordinate.
        name : str, optional
            The unique name of the coordinate.
        """
        
        self.dim: int = dim if dim is not None else len(components)
        self._components = Vertex(components, dim=dim)
        super().__init__(data=self._components.data, dim=dim, frame=frame, name=name)
    
    def _validate(self):
        """
        Validate that the data has the correct shape for a [`ComponentsBase`][pymagbot.math.geometry.ComponentsBase].

        Raises
        ------
        ValueError
            If the data shape is not equal to `(self.dim, )`.
        """
        if self.data.shape != (self.dim,):
            raise ValueError(f"ComponentsBase must have shape ({self.dim},), got {self.data.shape}.")
    
    @property
    def data(self) -> torch.Tensor:
        """Get the underlying [PyTorch](https://pytorch.org/) tensor data."""
        return self._components.data
    
    @data.setter
    def data(self, value: Union[torch.Tensor, ArrayLike, 'MatrixBase', list, tuple]):
        """
        Set the underlying [PyTorch](https://pytorch.org/) tensor data.

        Parameters
        ----------
        value : array_like
            The new data for the matrix.  It will be converted to a [PyTorch](https://pytorch.org/) tensor and moved to the appropriate device.
        """
        self._components.data = value
        self._data = self._initialize_data(self._components.data)
        # super().__init__(data=self._components.data, dim=self.dim, name=self.name, frame=self.frame)
    
    @property
    def components(self) -> Vertex:
        """Get the component data."""
        return self._components
    
    @components.setter
    def components(self, value: Union[torch.Tensor, ArrayLike, 'MatrixBase', list, tuple]):
        """
        Set the component data.

        Parameters
        ----------
        value : array_like
            The new coordinate's component.
        """
        self._components.data = Vertex(value)
        self._data = self._initialize_data(self._components.data)
    
    @property
    def x(self) -> torch.Tensor:
        """The x-component of the coordinate."""
        return self._components.data[0]
    
    @x.setter
    def x(self, value: Union[float, int]):
        """The x-component of the coordinate."""
        self._components.data[0] = value
        self._data = self._initialize_data(self._components.data)
    
    @property
    def y(self) -> torch.Tensor:
        """The x-component of the coordinate."""
        return self._components.data[1]
    
    @y.setter
    def y(self, value: Union[float, int]):
        """The x-component of the coordinate."""
        self._components.data[1] = value
        self._data = self._initialize_data(self._components.data)
    
    @property
    def z(self) -> torch.Tensor:
        """The x-component of the coordinate."""
        if self.dim < 3:
            raise ValueError("Coordinate is not 3D")
        return self._components.data[2]
    
    @z.setter
    def z(self, value: Union[float, int]):
        """The x-component of the coordinate."""
        if self.dim < 3:
            raise ValueError("Coordinate is not 3D")
        self._components.data[2] = value
        self._data = self._initialize_data(self._components.data)
    
    @property
    def express_in_local(self) -> Vertex:
        """The local components in the (local) `frame`."""
        return Vertex(self.data, dim=self.dim)
    
    @classmethod
    def from_dict(cls, dict_obj: Optional[Union[dict, str]], known: Optional[dict] = None) -> 'ComponentsBase':
        """ Create a [`Coordinate`][pymagbot.math.geometry.Coordinate] object from a dictionary.

        Parameters
        ----------
        dict_obj: dict or str
            The imported dictionary.
        known: dict, optional
            A dictionary of known frames (see also [`get_frame`][pymagbot.math.frame.get_frame])
        Returns
        -------
        Coordinate
            The Coordinate object.
        Raises
        ------
        TypeError
            If the `dict_obj` is not a Coordinate type.

        """
        _, data, name, frame = cls._from_dict(dict_obj, known)
        # data = dict_obj['coordinates'] if 'coordinates' in dict_obj else data
        return cls(components=data, frame=frame, name=name)
    
    def __repr__(self) -> str:
        """String representation of the object."""
        return (f"{self.__class__.__name__}(name='{self.name}', coordinates={self._components.array.tolist()}, "
                f"frame='{self.frame.name}')")


class Coordinate(ComponentsBase):
    """Represents a coordinate in a specific frame.
    
    A *coordinate* is a system that uses numbers to uniquely determine the position of a point in a specific
    [`frame`][pymagbot.math.frame.Frame].
    This is a thin wrapper around a [`Vertex`][pymagbot.math.spatial_primitives.Vertex], providing the
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
        geometry object whose properties (e.g., coordinates, vectors) are
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


class Span:
    """Represents a span of values for creating a mesh grid.
    
    Attributes:
    ----------
    start : float
        The starting name of the span.
    stop : float
        The ending name of the span.
    num : int
        The number of samples in the span.
    """
    
    def __init__(self, start: float = 0.0, stop: float = 1.0, num_samples: int = 1):
        """Initialize a Span.

        Parameters
        ----------
        start : float
            The starting name of the span.
        stop : float
            The ending name of the span.
        num_samples : int
            The number of samples in the span.
        """
        self.start: float = start
        self.stop: float = stop
        self.num: int = num_samples
    
    @property
    def step(self) -> float:
        """Spacing between values. """
        if self.num == 0:
            return 0.0
        return float((self.stop - self.start) / self.num)
    
    @step.setter
    def step(self, step):
        """Set number of samples to step. """
        self.num = int((self.stop - self.start) / step)
    
    def logspace(self, base: float = 10.0) -> torch.Tensor:
        """ The numbers spaced evenly on a log scale of this `Span`."""
        return torch.logspace(self.start, self.stop, self.num, base=base,
                              device=DEVICE, dtype=DTYPE)
    
    def linspace(self) -> torch.Tensor:
        """The linearly distributed numbers of this `Span`.

        Returns
        -------
        torch.Tensor
            A tensor containing the values in the span.
        """
        return torch.linspace(self.start, self.stop, self.num, device=DEVICE, dtype=DTYPE)
    
    to_tensor = linspace
    """Convert the span to a [PyTorch](https://pytorch.org/) tensor."""
    
    @property
    def data(self) -> torch.Tensor:
        """Get the underlying [PyTorch](https://pytorch.org/) tensor data."""
        return self.to_tensor()
    
    @classmethod
    def from_tensor(cls, tensor: Union[torch.Tensor, MatrixBase, list, tuple, ArrayLike]) -> 'Span':
        """Create a `Span` object from a tensor.

        Parameters
        ----------
        tensor : Union[torch.Tensor, list, tuple, np.ndarray]
            The tensor data.

        Returns
        -------
        Span
            The Span object.

        Raises
        ------
        ValueError
            If the tensor does not have at least three values.
        """
        if isinstance(tensor, MatrixBase):
            tensor = tensor.array
        elif isinstance(tensor, torch.Tensor):
            tensor = tensor.cpu().numpy()
        elif isinstance(tensor, (list, tuple)):
            tensor = np.array(tensor)
        if len(tensor) < 3:
            raise ValueError("Tensor must have at least three values to determine start, stop, and num_samples.")
        
        return cls(start=tensor[0], stop=tensor[-1], num_samples=len(tensor))
    
    def to_array(self) -> np.ndarray:
        """Convert the span to a [NumPy](https://numpy.org/) array.

        Returns
        -------
        np.ndarray
            An array containing the values in the span.
        """
        # return self.to_tensor().numpy()
        return np.linspace(self.start, self.stop, self.num)
    
    def __repr__(self) -> str:
        """String representation of the object."""
        return f"Span(start={self.start}, stop={self.stop}, num_samples={self.num})"
    
    @property
    def range(self) -> float:
        """Returns the `Span`'s range"""
        return self.stop - self.start
    
    @classmethod
    def from_range(cls, num_samples: int, range: float) -> 'Span':
        """Create a Span object from a number of samples and a range.

        Parameters
        ----------
        num_samples : int
            The number of samples.
        range : float
            The range of the span.

        Returns
        -------
        Span
            The Span object.
        """
        start = 0.0
        stop = range
        return cls(start=start, stop=stop, num_samples=num_samples)
    
    @classmethod
    def from_arange(cls, start: float, stop: float, step: float) -> 'Span':
        """Create a Span object from an [arange][numpy.arange].

        Parameters
        ----------
        start : float
            The starting name of the arange.
        stop : float
            The ending name of the arange.
        step : float
            The step size of the arange.

        Returns
        -------
        Span
            The Span object.
        """
        num_samples = int((stop - start) / step) + 1
        return cls(start=start, stop=stop, num_samples=num_samples)
    
    def to_dict(self) -> dict:
        """Converts the [`Span`][pymagbot.math.geometry.Span] object to a dictionary.

        Returns
        -------
        dict
            The object structure as a dictionary.
        """
        return {"start": self.start, "stop": self.stop, "num_samples": self.num}
    
    @classmethod
    def from_dict(cls, dict_obj: Union[dict, str]) -> 'Span':
        """Create a [`Span`][pymagbot.math.geometry.Span] object from a dictionary.

        Parameters
        ----------
        dict_obj : Union[dict, str]
            The imported dictionary.

        Returns
        -------
        Span
            The [`Span`][pymagbot.math.geometry.Span] object.

        Raises
        ------
        TypeError
            If the `dict_obj` is not a Span type.
        """
        dict_obj = parse_dict(dict_obj)
        start = dict_obj.get('start', 0)
        stop = dict_obj.get('stop', 0)
        num_samples = dict_obj.get('num_samples', 0)
        return cls(start=start, stop=stop, num_samples=num_samples)
    
    def __contains__(self, item: Union[tuple, list, int, float, torch.Tensor, ArrayLike, MatrixBase]):
        """
        Check if an item is within the span.
    
        Parameters
        ----------
        item : Union[tuple, list, int, float, torch.Tensor, MatrixBase]
            The item to check.
    
        Returns
        -------
        bool
            `True` if the item is within the span, `False` otherwise.
    
        Raises
        ------
        TypeError
            If the item type is unsupported.
        """
        if isinstance(item, (tuple, list, int, float, ArrayLike)):
            item = torch.as_tensor(item, device=DEVICE, dtype=DTYPE)
        elif isinstance(item, MatrixBase):
            item = item.data
        if not isinstance(item, torch.Tensor):
            raise TypeError('Unsupported `item` type.')
        return torch.logical_and(torch.all(self.start >= item), torch.all(item <= self.stop))


class Mesh(GeometryBase):
    """Represents a mesh grid of Cartesian coordinates in 2D or 3D space.
    
    A mesh  is a division of an interval (or [`Span`][pymagbot.math.geometry.Span]) that is a finite sequence of
    coordinates (e.g. evenly spaced) in a specific frame. The coordinates (or points) are arranged in a grid pattern.
    This is helpful for creating a grid of points in a specific frame or for a specific workspace.
    
    Attributes
    ----------
    grid : torch.Tensor
        The mesh grid points.
    spans : Sequence[Span]
        The spans of the mesh grid.
    dim: int
        The base dimension (2 or 3) of the geometric object.
    frame : Frame
        The reference frame of the geometry object. Default the [world frame][pymagbot.math.frame.Frame.world] is
        used.
    name : str
        The unique name of the geometry object.
    indexing: str
        The indexing mode, either “xy” or “ij”, defaults to “ij” (see also [torch.meshgrid][]).
    grid_x, grid_y, grid_z : torch.Tensor
        The grid coordinates in the x, y, and z directions.
    """
    
    grid_x, grid_y, grid_z = None, None, None
    """The grid coordinates in the x, y, and z directions."""
    
    def __init__(self, spans: Union[Sequence[Span], Span], dim: int = None, frame: Frame = None,
                 name: Optional[str] = None, indexing: str = 'ij'):
        """Initialize a Mesh.

        Parameters
        ----------
        spans : Union[Sequence[Span], Span]
            A sequence of [`Span`][pymagbot.math.geometry.Span] objects, one for each dimension,
            or a single span to be used for all dimensions.
        dim : int, optional
            The mesh dimension (2 or 3).
        frame : Frame, optional
            The reference frame of the mesh.
        name : str, optional
            The unique name of the mesh.
        indexing: str, optional
            The indexing mode, either “xy” or “ij”, defaults to “ij” (see also [torch.meshgrid][]).
        """
        if isinstance(spans, Span):
            if dim is None:
                dim = 3
            self._spans: Sequence[Span] = [spans] * dim
        else:
            self._spans: Sequence[Span] = spans
        self.dim: int = dim if dim is not None else len(self._spans)
        self.indexing: str = indexing
        if self.dim != len(self._spans):
            raise ValueError("The dim must be the same of the lengtj of spans")
        super().__init__(data=self._create_grid(self.indexing), dim=self.dim, frame=frame, name=name)
    
    def _validate(self):
        """Validate the shape and properties of the data """
        # if len(self.spans) != self.dim:
        #    raise ValueError(f"Number of spans ({len(self.spans)}) must match the mesh dimension ({self.dim}).")
        expected_shape = tuple(s.num for s in self.spans) + (self.dim,)
        if self.data.shape != expected_shape:
            raise ValueError(f"Mesh data shape {self.data.shape} does not match the expected shape {expected_shape}")
    
    def _create_grid(self, indexing: str = 'ij') -> torch.Tensor:
        """Create the mesh grid using [`torch.meshgrid`][torch.meshgrid].

        Returns
        -------
        torch.Tensor
            A tensor containing the coordinates of the mesh grid.
        """
        tensors = [span.linspace() for span in self.spans]
        if self.dim == 2:
            self.grid_x, self.grid_y = torch.meshgrid(*tensors, indexing=indexing)
            # Stack the coordinates to create the mesh grid
            grid = torch.stack((self.grid_x, self.grid_y), dim=-1)
        elif self.dim == 3:
            self.grid_x, self.grid_y, self.grid_z = torch.meshgrid(*tensors, indexing=indexing)
            grid = torch.stack((self.grid_x, self.grid_y, self.grid_z), dim=-1)
        else:
            raise ValueError("Mesh dimension must be 2 or 3.")
        return grid
    
    @property
    def grid(self) -> torch.Tensor:
        """The mesh grid points"""
        return self.data
    
    @grid.setter
    def grid(self, value: torch.Tensor):
        """Set new values to the grid points"""
        self.data = value
        self._create_coordinates()

    def _create_coordinates(self):
        """Create coordinates to easy manage the points.

        Check the dimension of the grid points
        """
        if self.dim == 2:
            self.grid_x, self.grid_y = self.grid.split([1, 1], dim=-1)
        elif self.dim == 3:
            self.grid_x, self.grid_y, self.grid_z = self.grid.split([1, 1, 1], dim=-1)
    
    @property
    def spans(self) -> Sequence[Span]:
        """Get the spans of the mesh grid."""
        return self._spans
    
    @spans.setter
    def spans(self, spans: Sequence[Span]):
        """Set the spans of the mesh grid."""
        if isinstance(spans, Span):
            self._spans: Sequence[Span] = [spans] * self.dim
        else:
            self._spans = spans
        if self.dim != len(self._spans):
            raise ValueError(f"The new `spans` value {len(self._spans)} and the `dim` {self.dim} are not the same.")
        self.data = self._create_grid(self.indexing)
    
    @property
    def x(self) -> torch.Tensor:
        """Get the x-coordinates of the mesh grid."""
        if self.dim < 1:
            raise ValueError("Mesh dimension must be at least 1.")
        return self.grid_x
    
    @x.setter
    def x(self, value: torch.Tensor):
        """Set the x-coordinates of the mesh grid."""
        if self.dim < 1:
            raise ValueError("Mesh dimension must be at least 1.")
        if value.shape != self.grid_x.shape:
            raise ValueError(f"X-coordinate shape {value.shape} does not match expected shape {self.grid_x.shape}")
        self.grid_x = value
        self.data = torch.stack((self.grid_x, self.grid_y), dim=-1) if self.dim == 2 else torch.stack(
            (self.grid_x, self.grid_y, self.grid_z), dim=-1)

    @property
    def y(self) -> torch.Tensor:
        """Get the y-coordinates of the mesh grid."""
        if self.dim < 2:
            raise ValueError("Mesh dimension must be at least 2.")
        return self.grid_y

    @y.setter
    def y(self, value: torch.Tensor):
        """Set the y-coordinates of the mesh grid."""
        if self.dim < 2:
            raise ValueError("Mesh dimension must be at least 2.")
        if value.shape != self.grid_y.shape:
            raise ValueError(f"Y-coordinate shape {value.shape} does not match expected shape {self.grid_y.shape}")
        self.grid_y = value
        self.data = torch.stack((self.grid_x, self.grid_y), dim=-1) if self.dim == 2 else torch.stack(
            (self.grid_x, self.grid_y, self.grid_z), dim=-1)
        
    @property
    def z(self) -> torch.Tensor:
        """Get the z-coordinates of the mesh grid."""
        if self.dim < 3:
            raise ValueError("Mesh dimension must be at least 3.")
        return self.grid_z
    
    @z.setter
    def z(self, value: torch.Tensor):
        """Set the z-coordinates of the mesh grid."""
        if self.dim < 3:
            raise ValueError("Mesh dimension must be at least 3.")
        if value.shape != self.grid_z.shape:
            raise ValueError(f"Z-coordinate shape {value.shape} does not match expected shape {self.grid_z.shape}")
        self.grid_z = value
        self.data = torch.stack((self.grid_x, self.grid_y, self.grid_z), dim=-1)
        
    @property
    def cartesian(self) -> tuple:
        """The Cartesian coordinates."""
        if self.dim == 2:
            return self.x, self.y
        elif self.dim == 3:
            return self.x, self.y, self.z
        else:
            raise ValueError("Mesh dimension must be 2 or 3.")
    
    @cartesian.setter
    def cartesian(self, coordinates: Union[tuple, list, torch.Tensor, ArrayLike]):
        """Set the Cartesian coordinates of the mesh grid.

        Parameters
        ----------
        coordinates : Union[tuple, list, torch.Tensor, ArrayLike]
            A tuple or list of coordinate tensors (x, y, z) or (x, y).
        """
        if not isinstance(coordinates, (tuple, list)):
            raise TypeError("Coordinates must be a tuple or list.")
        if len(coordinates) != self.dim:
            raise ValueError(f"Expected {self.dim} components, got {len(coordinates)}")
        
        # Ensure that coordinates are tensors and have the correct shape
        coordinates = [torch.as_tensor(c, dtype=DTYPE, device=DEVICE) for c in coordinates]
        expected_shape = self.data.shape[:-1]  # Shape of the grid without the dimension
        for c in coordinates:
            if c.shape != expected_shape:
                raise ValueError(f"Coordinate shape {c.shape} does not match expected shape {expected_shape}")
        # Update the grid
        if self.dim == 2:
            self.grid_x, self.grid_y = coordinates
            self.data = torch.stack((self.grid_x, self.grid_y), dim=-1)
        elif self.dim == 3:
            self.grid_x, self.grid_y, self.grid_z = coordinates
            self.data = torch.stack((self.grid_x, self.grid_y, self.grid_z), dim=-1)
        else:
            raise ValueError("Mesh dimension must be 2 or 3.")
        
    def resize(self, spans: Sequence[Span]):
        """Resize the mesh grid with new spans.

        Parameters
        ----------
        spans : Sequence[Span]
            The new spans.
        """
        self.spans = spans
    
    def express_in_frame(self, frame: Frame) -> 'Mesh':
        """Express this Mesh in a different `frame`.
        
        This method does not change the underlying geometry but returns a *new*
        geometry object whose properties (e.g., coordinates, vectors) are
        expressed with respect to the `target_frame`.

        Parameters
        ----------
        frame : Frame
            The frame in which to express the geometry.
        Returns
        -------
        Mesh
            A *new* Mesh object with its properties expressed in the `target_frame`.
        """
        if self.dim != frame.dim:
            raise ValueError(f"Mesh dimension ({self.dim}) and"
                             f"RigidTransform dimension ({frame.dim}) must match.")
        # Reshape the grid for transformation
        original_shape = self.data.shape
        reshaped_grid = self.data.reshape(-1, self.dim).T  # (dim, N)
        # Apply rotation and translation
        transformed_grid = (frame.local_transform.rotation.data @ reshaped_grid +
                            frame.local_transform.translation.data.reshape(-1, 1))  # (dim, N)
        new_grid = transformed_grid.T.reshape(original_shape)  # (N, dim)
        return Mesh(spans=self.spans, dim=self.dim, frame=frame, name=f"{self.name}_in_{frame.name}")
    
    def __repr__(self) -> str:
        """String representation of the object."""
        return f"Mesh(name='{self.name}', spans={self.spans}, shape={self.grid.shape}, frame='{self.frame.name}')"
    
    def to_dict(self, full: bool = True) -> dict:
        """Converts the [`Mesh`][pymagbot.math.geometry.Mesh] object to a dictionary.

        Parameters
        ----------
        full : bool, optional
            Whether to export all the structure or only the relevant parts. Defaults to `True`.
        Returns
        -------
        dict
            The object structure as a dictionary.
        """
        dict_obj: dict = super().to_dict(full=False)
        dict_obj['spans'] = [span.to_dict() for span in self.spans]
        return dict_obj
    
    @classmethod
    def from_dict(cls, dict_obj: Union[dict, str], known: Optional[dict] = None) -> 'Mesh':
        """Create a [`Mesh`][pymagbot.math.geometry.Mesh] object from a dictionary.

        Parameters
        ----------
        dict_obj : Union[dict, str]
            The imported dictionary.
        known : dict, optional
            A dictionary of known frames.

        Returns
        -------
        Mesh
            The [`Mesh`][pymagbot.math.geometry.Mesh] object.

        Raises
        ------
        TypeError
            If the `dict_obj` is not a Mesh type.
        """
        _, data, name, frame = cls._from_dict(dict_obj, known)
        spans: Sequence[Span] = [Span.from_dict(span) for span in data]
        return cls(spans=spans, dim=len(spans), name=name, frame=frame)
    
    @classmethod
    def from_dataframe(cls, data):
        """Create a `Mesh` from a [`pandas.DataFrame`][pandas.DataFrame]."""
        import pandas as pd
        if not isinstance(data, pd.DataFrame):
            raise TypeError('data should be a `pandas.DataFrame`')
        data.columns = data.columns.str.replace(' ', '').str.lower()
        if not all([x in data.columns for x in ('vx', 'vy', 'vz')]):
            raise ValueError('Cannot use this `pandas.DataFrame` data to make a `Mesh`')
        
        span_x = Span(min(data.x), max(data.x), len(data.x))
        span_y = Span(min(data.y), max(data.y), len(data.x))
        span_z = Span(min(data.z), max(data.z), len(data.x))
        return cls((span_x, span_y, span_z))


class EuclideanVector(ComponentsBase):
    """Represents a Euclidean vector with magnitude and direction.
    
    A Euclidean vector, or simply a _vector_, is a geometric object that has both **magnitude** and **direction**. It is
    represented by a point in space and a direction.
    This is a thin wrapper around a [`Vertex`][pymagbot.math.spatial_primitives.Vertex], providing
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
        geometry object whose properties (e.g., coordinates, vectors) are
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


class VectorField(GeometryBase):
    """Represents a vector field in 2D or 3D space.

    A vector field is an assignment of a vector to each point in space.  Internally,
    the [`VectorField`][pymagbot.math.geometry.VectorField] stores the vector
    (i.e. [`EuclideanVector`][pymagbot.math.geometry.EuclideanVector]) components at each grid point of a
    [`Mesh`][pymagbot.math.geometry.Mesh].
    
    Examples:
    --------
    ```python
    import torch
    from pymagbot.math.frame import Frame
    from pymagbot.math.geometry import VectorField, Mesh, Span
    
    #Define the grid and the vectors
    span_x = Span(0, 1, 10)
    span_y = Span(0, 1, 10)
    mesh = Mesh([span_x, span_y])
    F_x = mesh.y * np.exp(mesh.x)
    F_y = mesh.x**2 + np.exp(mesh.x)
    F_z = mesh.z**2 * np.exp(mesh.z)
    vector_field = VectorField(vectors = vectors, points=mesh, frame=Frame.world)
    print(vector_field)
    ```
    
    Attributes
    ----------
    vectors : torch.Tensor
        The vector components at each point in the mesh.
    mesh : Mesh
        The mesh grid defining the points in space where the vectors are defined.
    dim: int
        The base dimension (2 or 3) of the geometric object.
    frame : Frame
        The reference frame of the geometry object. Default the [world frame][pymagbot.math.frame.Frame.world] is
        used.
    name : str
        The unique name of the geometry object.
    """
    
    # vectors = super().data  # Ensure that the vector components should be stored in _data attribute from MatrixBase
    
    def __init__(self, vectors: Union[ArrayLike, torch.Tensor, list, tuple], mesh: Mesh, dim: int = None,
                 frame: Frame = None,
                 name: Optional[str] = None):
        """Initialize a VectorField.

        Parameters
        ----------
        vectors : torch.Tensor
            Vector components at each point in the mesh.
        mesh : Mesh, optional
            The mesh grid defining the points in space where the vectors are defined.
            For example, if [mesh.grid.shape] is (10, 10, 3), then vectors.shape should be (10, 10, 3).
        name : str, optional
            The unique name of the vector field.
        """
        if mesh is None:
            dim = vectors.shape[-1]  # infer the dimension from the shape of vector
            spans: List[Span] = []
            for k in range(dim - 1):
                # Set the default values for new spans dimension to set the new default mesh
                span = Span(start=0, stop=1, num_samples=vectors.shape[k])
                spans.append(span)
                mesh = Mesh(spans=spans, dim=dim)
        self.mesh: Mesh = mesh
        super().__init__(data=vectors, dim=dim, frame=frame, name=name)
    
    def _validate(self):
        """Validates that the `vectors` tensor has the correct shape.

        Raises
        ------
        ValueError
            If the shape of `vectors` is incompatible with the shape of the `mesh`.
        """
        if self.mesh.dim != self.vectors.shape[-1]:
            raise ValueError("Vector must have the same dim as the dimension in its parameters' space/mesh")
        expected_shape = (*self.mesh.grid.shape[:-1], self.mesh.dim)
        if self.vectors.shape != expected_shape:
            raise ValueError(f"The shape of 'vectors' must be {expected_shape}, but got {self.vectors.shape}")
    
    @property
    def vectors(self) -> torch.Tensor:
        """Get the vector data"""
        return self.data
    
    @vectors.setter
    def vectors(self, vectors: torch.Tensor):
        """Set the  vector  data"""
        self.data = vectors
    
    def express_in_frame(self, frame: Frame):
        """Transform the vector field to the provided rigid transformation.
        
        The mesh is transformed first, then each vector in the field is rotated
        accordingly.
        """
        if self.mesh.dim != frame.dim:
            raise ValueError(
                f"Mesh dimension ({self.mesh.dim}) and RigidTransform dimension ({frame.dim}) must match.")
        
        self.mesh.express_in_frame(frame)
        # Apply rotation to each vector in the field
        original_shape = self.vectors.shape
        reshaped_vectors = self.vectors.reshape(-1, self.mesh.dim).T  # (dim, N)
        rotated_vectors = frame.local_rotation.data @ reshaped_vectors  # (dim, N)
        self.vectors = rotated_vectors.T.reshape(original_shape)
    
    def __repr__(self) -> str:
        """String representation of the object."""
        return f"VectorField(name='{self.name}', vectors.shape={self.vectors.shape}, mesh={self.mesh}," \
               f" frame='{self.frame.name}')"
    
    def to_dict(self, full: bool = True) -> dict:
        """Converts the [`VectorField`][pymagbot.math.geometry.VectorField] object to a dictionary.

        Parameters
        ----------
        full : bool, optional
            Whether to export all the structure or only the relevant parts. Defaults to `True`.

        Returns
        -------
        dict
            The object structure as a dictionary.
        """
        dict_obj: dict = super().to_dict(full=False)
        if full:
            dict_obj['vectors'] = self.vectors.tolist()
            dict_obj['mesh'] = self.mesh.to_list()
        return dict_obj
    
    @classmethod
    def from_dict(cls, dict_obj: Union[dict, str], known: Optional[dict] = None) -> 'VectorField':
        """Create a [`VectorField`][pymagbot.math.geometry.VectorField] object from a dictionary.

        Parameters
        ----------
        dict_obj : Union[dict, str]
            The imported dictionary.
        known : dict, optional
            A dictionary of known frames.

        Returns
        -------
        VectorField
            The [`VectorField`][pymagbot.math.geometry.VectorField] object.

        Raises
        ------
        TypeError
            If the `dict_obj` is not a VectorField type.
        """
        _, data, name, frame = cls._from_dict(dict_obj, known)
        vectors = torch.as_tensor(dict_obj['vectors'], dtype=DTYPE, device=DEVICE)
        mesh = Mesh.from_dict(dict_obj['mesh'], known)
        return cls(vectors=vectors, mesh=mesh, name=name, frame=frame)
