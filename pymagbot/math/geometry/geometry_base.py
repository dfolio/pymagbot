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
"""The `geometry` submodule to manage objects that are attached to a [`Frame`][pymagbot.math.frame.Frame]."""

import itertools
import weakref
from abc import ABC, abstractmethod
from typing import Optional, Union

import torch
from numpy.typing import ArrayLike

from pymagbot.math.matrix_base import MatrixBase, parse_dict
from pymagbot.math.spatial.frame import Frame, get_frame
from pymagbot.math.spatial.primitives import Vertex
from pymagbot.utils.config import DTYPE, USE_CUDA

__all__ = ['GeometryBase', 'ComponentsBase']

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
        geometry object whose properties (e.g., coordinates, components) are
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
        geometry object whose properties (e.g., coordinates, components) are
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
                 dim: int = None, system: str = 'cartesian', frame: Frame = None, name: Optional[str] = None):
        """Initialize a [`ComponentsBase`][pymagbot.geometry.ComponentsBase].

        Parameters
        ----------
        components : Union[Vertex, ArrayLike, torch.Tensor, list, tuple]
            The coordinates of the point.
        dim : int, optional
            The dimension of the coordinate (2 or 3).
        system : str, optional
            The coordinate system of the component (cartesian, spherical, or cylindrical).
        frame : Frame, optional
            The reference frame of the coordinate.
        name : str, optional
            The unique name of the coordinate.
        """
        
        self.dim: int = dim if dim is not None else len(components)
        self._components = Vertex(components, dim=dim, system=system)
        self._system = self._components.system
        super().__init__(data=self._components.data, dim=dim, frame=frame, name=name)
    
    def __repr__(self) -> str:
        """String representation of the object."""
        return (f"{self.__class__.__name__}(name='{self.name}', components={self._components.array.tolist()}, "
                f"frame='{self.frame.name}')")
    
    def express_in_local(self) -> 'ComponentsBase':
        """The components expressed in the (local)  [`Frame`][pymagbot.math.spatial.frame.Frame]."""
        return self
    
    def to_cartesian(self) -> 'ComponentsBase':
        """Convert the components to Cartesian coordinates."""
        return self.__class__(components=self.components.to_cartesian(), dim=self.dim, frame=self.frame,
                              name=f"{self.name}_cartesian")
    
    def to_polar(self) -> 'ComponentsBase':
        """Convert the components to Polar coordinates."""
        return self.__class__(components=self.components.to_polar(), dim=2,  # Polar exist only in 2D
                              frame=self.frame, name=f"{self.name}_polar")
    
    def to_cylindrical(self) -> 'ComponentsBase':
        """Convert the components to Cylindrical coordinates."""
        return self.__class__(components=self.components.to_cylindrical(), dim=self.dim, frame=self.frame,
                              name=f"{self.name}_cylindrical")
    
    def to_spherical(self) -> 'ComponentsBase':
        """Convert the components to Spherical coordinates."""
        return self.__class__(components=self.components.to_spherical(), dim=self.dim, frame=self.frame,
                              name=f"{self.name}_spherical")
    
    @property
    def data(self) -> torch.Tensor:
        """Get the underlying [PyTorch](https://pytorch.org/) tensor data."""
        return self._components.data
    
    @data.setter
    def data(self, value: Union[torch.Tensor, ArrayLike, 'MatrixBase', list, tuple]):
        """Set the underlying [PyTorch](https://pytorch.org/) tensor data.

        Parameters
        ----------
        value : array_like
            The new data for the matrix.  It will be converted to a [PyTorch](https://pytorch.org/) tensor and
            moved to the appropriate device.
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
        """Set the component data.

        Parameters
        ----------
        value : array_like
            The new coordinate's component.
        """
        self._components.data = Vertex(value, dim=self.dim, system=self._system)
        self._data = self._initialize_data(self._components.data)
    
    @property
    def system(self) -> str:
        """Get the coordinate system of the vector."""
        return self._system
    
    @system.setter
    def system(self, new_system: str):
        new_system = new_system.lower()
        if new_system not in Vertex.allowed_systems:
            raise ValueError(f"system must be one of {Vertex.allowed_systems}, got {new_system}")
        self._system = new_system
        self.data = self.components.to_system(self._system).components.data
        
    @property
    def x(self) -> torch.Tensor:
        """The x-component of the `ComponentsBase`."""
        return self.components.x
    
    @x.setter
    def x(self, value:  Union[float, int, torch.Tensor]):
        """The x-component of the `ComponentsBase`."""
        self.components.x = value
        self._data = self._initialize_data(self._components.data)
    
    @property
    def y(self) -> torch.Tensor:
        """The x-component of the `ComponentsBase`."""
        return self.components.y
    
    @y.setter
    def y(self, value:  Union[float, int, torch.Tensor]):
        """The x-component of the `ComponentsBase`."""
        self.components.y = value
        self._data = self._initialize_data(self._components.data)
    
    @property
    def z(self) -> torch.Tensor:
        """The z-component of the `ComponentsBase`."""
        return self.components.z
    
    @z.setter
    def z(self, value:  Union[float, int, torch.Tensor]):
        """The z-component of the `ComponentsBase`."""
        self.components.z = value
        self._data = self._initialize_data(self._components.data)
    
    @property
    def rho(self) -> torch.Tensor:
        """The rho-component of the `ComponentsBase`."""
        return self.components.rho
    
    @rho.setter
    def rho(self, value:  Union[float, int, torch.Tensor]):
        """The rho-component of the `ComponentsBase`."""
        self.components.rho = value
        self._data = self._initialize_data(self._components.data)
    
    @property
    def phi(self) -> torch.Tensor:
        """The phi-component of the `ComponentsBase`."""
        return self.components.phi
    
    @phi.setter
    def phi(self, value:  Union[float, int, torch.Tensor]):
        """The phi-component of the `ComponentsBase`."""
        self.components.phi = value
        self._data = self._initialize_data(self._components.data)
    
    @property
    def theta(self) -> torch.Tensor:
        """The theta-component of the `ComponentsBase`."""
        return self.components.theta
    
    @theta.setter
    def theta(self, value:  Union[float, int, torch.Tensor]):
        """The theta-component of the `ComponentsBase`."""
        self.components.theta = value
        self._data = self._initialize_data(self._components.data)
    
    def to_dict(self, full: bool = True) -> dict:
        """Converts the`ComponentsBase` object to a dictionary.

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
        dict_obj['components'] = self.components.to_dict()
        return dict_obj
    
    @classmethod
    def from_dict(cls, dict_obj: Optional[Union[dict, str]], known: Optional[dict] = None) -> 'ComponentsBase':
        """ Create a`ComponentsBase` object from a dictionary.

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
            If the `dict_obj` is not a ComponentsBase type.
        """
        _, _, name, frame = cls._from_dict(dict_obj, known)
        _components = Vertex.from_dict(dict_obj["components"]) if "components" in dict_obj else None
        return cls(components=_components, frame=frame, name=name)
    
    def _validate(self):
        """
        Validate that the data has the correct shape for a`ComponentsBase`.

        Raises
        ------
        ValueError
            If the data shape is not equal to `(self.dim, )`.
        """
        if self.data.shape != (self.dim,):
            raise ValueError(f"ComponentsBase must have shape ({self.dim},), got {self.data.shape}.")
