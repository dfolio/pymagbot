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
spatial_primitives - Submodule for base spatial primitives.

This module defines spatial primitives for `pymagbot`, including vector-like objects such as
[`VectorBase`][pymagbot.math.spatial.primitives.VectorBase], [`Vertex`][pymagbot.math.spatial.primitives.Vertex],
and [`Direction`][pymagbot.math.spatial.primitives.Direction]. These classes provide fundamental representations
and operations for handling spatial transformations and geometry in micro/nano-robotics.

Notes
-----
TODO: Future implementations may include additional spatial primitives like `Line` and `Plane`.
"""

from typing import Optional, Union

import torch
from numpy.typing import ArrayLike
from pyparsing import lineStart

# from pymagbot.math.geometry import ComponentsBase
from pymagbot.math.matrix_base import ensure_tensor, MatrixBase
from pymagbot.math.spatial.operations import check_as_tensor, matrix_is_orthogonal
from pymagbot.utils.config import DEVICE, DTYPE

__all__ = ['VectorBase', 'Vertex', 'Direction']


# Define helper function at module level to avoid redundancy
def _validate_system(system: str, allowed_systems: set) -> str:
    """Validate if a given system is within the allowed systems."""
    system = system.lower()
    if system not in allowed_systems:
        raise ValueError(f"System must be one of {allowed_systems}, got {system}")
    return system
    
    
class VectorBase(MatrixBase):
    """Base class for vector-like primitives.
    
    A vector is a tuple of $n$ (here $n=2$ or 3) real numbers has a natural structure of vector space defined by
    component-wise addition and scalar multiplication.
    This class also provides common operations for vectors, including coordinate system conversions.
    
    Notes
    -----
    All angles are expressed in radians (unless otherwise noted).
    
    Attributes
    ----------
    components : array_like, optional
        The components of vector.
    dim : int, optional
        The dimension of the vector (e.g., 2D or 3D)
    system : str, optional
        The coordinate system of the vector. Allowed values are 'cartesian', 'cylindrical',
        'spherical', or 'polar'. Default is 'cartesian'.
    """
    
    # Allowed coordinate systems
    allowed_systems = {'cartesian', 'cylindrical', 'spherical', 'polar'}
    _systems: str = 'cartesian'
    
    def __init__(self, components: Optional[Union[torch.Tensor, ArrayLike, 'VectorBase', list, tuple]] = None,
                 dim: int = None, system: str = 'cartesian'):
        
        _validate_system(system, self.allowed_systems)
        super().__init__(components, dim=dim)
        self._system = system  # Use the private attribute directly
    
    def __repr__(self) -> str:
        """String representation of the object."""
        return f"{self.__class__.__name__}(components={self.array.tolist()})"
    
    def distance(self, other: 'VectorBase') -> torch.Tensor:
        """Compute the Euclidean distance between two components.

        Parameters
        ----------
        other : VectorBase
            The other vector to compute the distance to.

        Returns
        -------
        torch.Tensor
            The Euclidean distance between the two components.
        """
        other = ensure_tensor(other, device=self.device)
        return torch.linalg.norm(self.data - other.data)
    
    def to_list(self) -> list:
        """Convert the vector-like object to a list."""
        return [x for x in self.data]
    
    def to_cartesian(self) -> 'VectorBase':
        """Convert the vector to Cartesian coordinates.

        Returns
        -------
        VectorBase
            A new instance with Cartesian coordinates (x, y, z).
        """
        if self.system == 'cartesian':
            return self
        
        if self.dim == 2:
            if self.system == 'polar':
                r,  phi = self.to_list()
                x = r * torch.cos(phi)
                y = r * torch.sin(phi)
                return self.__class__([x, y], dim=2, system='cartesian')
            else:
                raise NotImplementedError(f"Conversion from {self.system} to cartesian not supported")
        elif self.dim == 3:
            if self.system == 'cylindrical':
                rho,  phi, z = self.to_list()
                x = rho * torch.cos(phi)
                y = rho * torch.sin(phi)
                return self.__class__([x, y, z], dim=3, system='cartesian')
            elif self.system == 'spherical':
                r,  phi, theta = self.to_list()
                x = r * torch.sin(theta) * torch.cos(phi)
                y = r * torch.sin(theta) * torch.sin(phi)
                z = r * torch.cos(theta)
                return self.__class__([x, y, z], dim=3, system='cartesian')
            else:
                raise NotImplementedError(f"Conversion from {self.system} to cartesian not supported")
        else:
            raise ValueError("to_cartesian() only supports 2D and 3D vectors.")
    
    def to_polar(self) -> 'VectorBase':
        """Convert a 2D vector to polar coordinates.

        Returns
        -------
        VectorBase
            A new instance with polar coordinates (r, phi), where r is the magnitude and phi is the angle.
        """
        if self.dim != 2:
            raise ValueError("to_polar() only supports 2D vectors.")
        if self.system == 'polar':
            return self
        if self.system == 'cartesian':
            x, y = self.to_list()
            r = self.magnitude
            phi = torch.atan2(y, x)
            return self.__class__([r, phi], dim=2, system='polar')
        else:
            raise NotImplementedError(f"Conversion from {self.system} to polar not supported.")
    
    def to_cylindrical(self) -> 'VectorBase':
        """Convert the vector to cylindrical coordinates.

        For a 3D vector, the cylindrical coordinates are defined as:
            (r, phi, z)
        where:
            - r is the radial component to the z-axis,
            - phi is the azimuthal angle (in radians),
            - z is the vertical component.

        For a 2D vector, z is omitted.

        Returns
        -------
        VectorBase
            The cylindrical components (r, phi, z) for 3D, or (r, phi, 0) for 2D.
        """
        if self.system in ['cylindrical', 'polar']:
            return self
        if self.dim == 2:
            polar_vec = self.to_polar()
            return self.__class__(polar_vec.components, dim=self.dim, system='cylindrical')
        elif self.dim == 3:
            if self.system == 'cartesian':
                rho = torch.sqrt(self.x**2 + self.y**2)
                phi = torch.atan2(self.y, self.x)
                return self.__class__([rho, phi, self.z], dim=self.dim, system='cylindrical')
            elif self.system == 'spherical':
                rho = self.r * torch.sin(self.theta)
                z = self.r * torch.cos(self.theta)
                return self.__class__([rho, self.phi, z], dim=self.dim, system='cylindrical')
            else:
                raise NotImplementedError(f"Conversion from {self.system} to cylindrical not supported")
        else:
            raise ValueError("to_cylindrical() only supports 2D or 3D vectors.")
    
    def to_spherical(self) -> 'VectorBase':
        """Convert the vector to spherical coordinates.

        For a 2D vector, spherical conversion is equivalent to polar representation.
        For a 3D vector, returns (r, phi, theta), with
        - r is the radial or radius distance component
        - phi is the azimuthal angle component (in radians).
        - theta is the polar angle component (in radians),

        Returns
        -------
        VectorBase
            A new instance with the vector in spherical (or polar for 2D) coordinates.
        """
        if self.system == 'spherical':
            return self
        
        if self.dim == 2:
            return self.to_polar()
        elif self.dim == 3:
            if self.system == 'cartesian':
                r = self.magnitude
                theta = torch.acos(self.z / r)
                phi = torch.atan2(self.y, self.x)
                return self.__class__([r, phi, theta], dim=3, system='spherical')
            elif self.system == 'cylindrical':
                r = self.r
                theta = torch.atan2(self.rho, self.z)
                phi = self.phi
                return self.__class__([r, phi, theta], dim=3, system='spherical')
            else:
                raise NotImplementedError(f"Conversion from {self.system} to spherical not supported")
        else:
            raise ValueError("to_spherical() only supports 2D and 3D vectors.")
    
    def to_system(self, system: str) -> 'VectorBase':
        """Helper function to convert the vector to another system coordinates."""
        if system == 'cartesian':
            return self.to_cartesian()
        if system == 'polar' and self.dim == 2:
            return self.to_polar()
        if system == 'cylindrical':
            return self.to_cylindrical()
        if system == 'spherical':
            return self.to_spherical()
        raise ValueError(f"Conversion from {self.system} to {system} not supported.")
    
    def apply_rigid_transform(self, rotation_matrix:Union[torch.Tensor, ArrayLike, 'MatrixBase', list, tuple],
                              translation_vector: Union[torch.Tensor, ArrayLike, 'VectorBase', list, tuple]) -> \
            'VectorBase':
        """Helper to apply a rigid transformation to the vector.

        Applies a rotation followed by a translation, optimizing tensor operations.

        Parameters
        ----------
        rotation_matrix : array_like
            A rotation matrix.
        translation_vector : torch.Tensor
            A translation vector.

        Returns
        -------
        VectorBase
            A new VectorBase object with the transformed coordinates.
        """
        if self.dim not in (2, 3):
            raise ValueError("Rigid transformation is only supported for 2D or 3D vectors.")
        rotation_matrix = check_as_tensor(rotation_matrix)
        if not matrix_is_orthogonal(rotation_matrix):
            raise ValueError("Rotation matrix is not orthogonal.")
        translation_vector = check_as_tensor(translation_vector)
        if rotation_matrix.shape[0] != self.dim or translation_vector.shape[0] != self.dim:
            raise ValueError("Dimension mismatch.")
        
        cartesian_vector = self.to_cartesian()
        components = cartesian_vector.components
        # Ensure correct dimensions for matrix multiplication
        if components.ndim == 1:
            components = components.unsqueeze(0)  # Reshape to (1, dim) row vector
        
        transformed_components = (rotation_matrix @ components.T).T + translation_vector
        transformed_cartesian = self.__class__(transformed_components, dim=self.dim, system='cartesian')
        return transformed_cartesian.to_system(self.system)
    
    def clone(self) -> 'VectorBase':
        """
        Create a deep copy of the vector.

        Returns
        -------
        VectorBase
            A new instance with the same data and coordinate system.
        """
        # openai: Assumes the subclass __class__ is appropriate
        return self.__class__(self.data.clone(), dim=self.dim, system=self.system)
    
    def to_dict(self):
        return {"components": self.array.tolist(), "system": self.system}
    
    @classmethod
    def from_dict(cls, data: dict) -> 'VectorBase':
        return cls(data["components"], system=data.get("system", "cartesian"))
    
    @property
    def system(self) -> str:
        """Get the coordinate system of the vector."""
        return self._system
    
    @system.setter
    def system(self, new_system: str):
        self._system = _validate_system(new_system, self.allowed_systems)
        self.data = self.to_system(self._system).components.data
    
    @property
    def components(self) -> torch.Tensor:
        """The [`VectorBase`][pymagbot.math.spatial.primitives.VectorBase] components. """
        return self._data.squeeze()
    
    @components.setter
    def components(self, value: Union[torch.Tensor, ArrayLike, 'VectorBase', list, tuple]):
        """Setter for the components."""
        self.data = value
    
    @property
    def x(self) -> torch.Tensor:
        """Get the x-component of the vector Cartesian."""
        if self.system != 'cartesian':
            raise AttributeError("x-component is only valid for Cartesian system.")
        return self.components[0]
    
    @x.setter
    def x(self, value:  Union[float, int, torch.Tensor]):
        """Set the x-component of the vector Cartesian."""
        if self.system != 'cartesian':
            raise AttributeError("x-component is only valid for Cartesian system.")
        self._set_component(0, value)
    
    @property
    def y(self) -> torch.Tensor:
        """Get the y-component of the vector (Cartesian)."""
        if self.system != 'cartesian':
            raise AttributeError("y-component is only valid for Cartesian system.")
        if self.dim < 2:
            raise AttributeError("y-component is only valid for 2D or 3D vectors.")
        return self.components[1]
    
    @y.setter
    def y(self, value:  Union[float, int, torch.Tensor]):
        """Set the y-component of the vector (Cartesian)."""
        if self.system != 'cartesian':
            raise AttributeError("y-component is only valid for Cartesian system.")
        if self.dim < 2:
            raise AttributeError("y-component is only valid for 2D or 3D vectors.")
        self._set_component(1, value)
    
    @property
    def z(self) -> torch.Tensor:
        """Get the z-component of the vector (Cartesian or Cylindrical)."""
        if self.system not in ('cartesian', 'cylindrical'):
            raise AttributeError("z-component is only valid for Cartesian and Cylindrical systems.")
        if self.dim < 3:
            raise AttributeError("z-component is only valid for 3D vectors.")
        return self.components[2]
    
    @z.setter
    def z(self, value:  Union[float, int, torch.Tensor]):
        """Set the z-component of the vector (Cartesian or Cylindrical)."""
        if self.system not in ('cartesian', 'cylindrical'):
            raise AttributeError("z-component is only valid for Cartesian and Cylindrical systems.")
        if self.dim < 3:
            raise AttributeError("z-component is only valid for 3D vectors.")
        self._set_component(2, value)
    
    @property
    def rho(self) -> torch.Tensor:
        """Get the rho (radial distance from the z-axi)  component of the vector (Cylindrical)."""
        if self.system != 'cylindrical':
            raise AttributeError("rho-component is only valid for Cylindrical system.")
        if self.dim < 2:
            raise AttributeError("rho-component is only valid for 2D or 3D vectors.")
        return self.components[0]
    
    @rho.setter
    def rho(self, value: Union[float, int, torch.Tensor]):
        """Set the rho (radial distance from the z-axi) component of the vector (Cylindrical)."""
        if self.system != 'cylindrical':
            raise AttributeError("rho-component is only valid for Cylindrical system.")
        if self.dim < 2:
            raise AttributeError("rho-component is only valid for 2D or 3D vectors.")
        self._set_component(0, value)
    
    @property
    def phi(self) -> torch.Tensor:
        """Get the phi component of the vector (Cylindrical or Spherical or Polar)."""
        if self.system not in ('cylindrical', 'spherical', 'polar'):
            raise AttributeError("phi-component is only valid for Cylindrical, Spherical and Polar systems.")
        if self.dim < 2:
            raise AttributeError("phi-component is only valid for 2D or 3D vectors.")
        return self.components[1]
    
    @phi.setter
    def phi(self, value:  Union[float, int, torch.Tensor]):
        """Set the phi component of the vector (Cylindrical or Spherical or Polar)."""
        if self.system not in ('cylindrical', 'spherical', 'polar'):
            raise AttributeError("phi-component is only valid for Cylindrical, Spherical and Polar systems.")
        if self.dim < 2:
            raise AttributeError("phi-component is only valid for 2D or 3D vectors.")
        self._set_component(1, value)
    
    @property
    def theta(self) -> torch.Tensor:
        """Get the theta component of the vector (Spherical)."""
        if self.system != 'spherical':
            raise AttributeError("theta-component is only valid for Spherical system.")
        if self.dim < 3:
            raise AttributeError("theta-component is only valid for 3D vectors.")
        return self.components[2]
    
    @theta.setter
    def theta(self, value:  Union[float, int, torch.Tensor]):
        """Set the theta component of the vector (Spherical)."""
        if self.system != 'spherical':
            raise AttributeError("theta-component is only valid for Spherical system.")
        if self.dim < 3:
            raise AttributeError("theta-component is only valid for 3D vectors.")
        self._set_component(2, value)
    
    @property
    def r(self) -> torch.Tensor:
        """Get the r (radius) component of the vector (Spherical or Polar)."""
        if self.system not in ('spherical', 'polar'):
            raise AttributeError("r-component is only valid for Spherical and Polar systems.")
        return self.components[0]
    
    @r.setter
    def r(self, value:  Union[float, int, torch.Tensor]):
        """Set the r (radius) component of the vector (Spherical or Polar)."""
        if self.system not in ('spherical', 'polar'):
            raise AttributeError("r-component is only valid for Spherical and Polar systems.")
        self._set_component(0, value)
    
    @property
    def magnitude(self) -> torch.Tensor:
        """Get the magnitude  (i.e. Euclidean norm) of the vector."""
        if self.system == 'cartesian':
            return torch.linalg.vector_norm(self.components, ord=2)
        if self.system in ('spherical', 'polar'):
            return self.r
        if self.system == 'cylindrical':
            return torch.sqrt(self.rho**2 + self.z**2)
        raise AttributeError("magnitude cannot be computed for this system.")
    
    @magnitude.setter
    def magnitude(self, mag):
        """The vector magnitude (Euclidean norm)"""
        self.scale(mag / self.magnitude)
    
    def _initialize_data(self, components: Optional[Union[torch.Tensor, ArrayLike, 'VectorBase', list, tuple]] = None):
        """Initialize a vecto-like.

        Parameters
        ----------
        components : array_like, optional
            The coordinates to initialize the vector with.

        Returns
        -------
        torch.Tensor
            The initialized tensor representation of the vector.
        """
        if components is None:
            # self.dim = self.dim if self.dim is not None else 3
            return torch.zeros((self.dim,), dtype=DTYPE, device=self.device)
        if isinstance(components, VectorBase):
            data = components.data.clone()
        elif isinstance(components, torch.Tensor):
            data = components.clone().to(DTYPE).to(self.device)
        else:
            data = torch.as_tensor(components, dtype=DTYPE, device=self.device)
            if self.dim is None:
                self.dim = len(components)  # Infer dimension from the data if not specified
        return data
    
    def _validate(self):
        """Validate that the vector-like has the correct shape.

        Raises
        ------
        ValueError
            If the vector-like shape is not equal to `(self.dim,)`.
        """
        if self.data.shape != (self.dim,):
            raise ValueError(f"Vector must have shape ({self.dim},), got {self.data.shape}.")
    
    def _set_component(self, index: int, value:  Union[float, int, torch.Tensor]):
        """Set a component of the vector and maintain tensor properties."""
        if not isinstance(value, (float, int, torch.Tensor)):
            raise TypeError("Component value must be a float, int, or torch.Tensor.")
        
        value = torch.as_tensor(value, dtype=DTYPE, device=self.device)
        if value.numel() != 1:  # Ensure it's a scalar value
            raise ValueError("Component value must be a scalar.")
        
        new_components = self.components.clone().detach()  # Clone and detach to avoid modifying the original tensor
        new_components[index] = value
        self.components = new_components  # Use the components setter to properly update the data
    
    
class Vertex(VectorBase):
    """A  vertex (or point) representation in 2D or 3D space.

    In geometry, a vertex is like a point.
    Specifically, a vertex is where two or more curves, lines or edges intersect or meet.
    
    Attributes
    ----------
    components : torch.Tensor
        The vertex components
    dim : int
        The dimension of the rotation matrix (2 or 3).
    system : str, optional
        The coordinate system of the vector. Allowed values are 'cartesian', 'cylindrical',
        'spherical', or 'polar'. Default is 'cartesian'.
    """
    
    def homogeneous(self) -> torch.Tensor:
        """Convert the vertex to homogeneous coordinates.
        
        Returns
        -------
        torch.Tensor
            Homogeneous coordinates representation.
        """
        return torch.cat((self.data, torch.tensor([1.], dtype=DTYPE, device=self.device)))
    
    @classmethod
    def from_homogeneous(cls, data: Union[torch.Tensor, ArrayLike, list, tuple]) -> 'Vertex':
        """Create a vertex from homogeneous coordinates.
        
        Parameters
        ----------
        data : torch.Tensor
            The homogeneous coordinates.
        
        Returns
        -------
        Vertex
            The vertex representation.
        """
        if not isinstance(data, torch.Tensor):
            data = torch.as_tensor(data, dtype=DTYPE, device=DEVICE)
        return cls(data[:-1])
    
    @classmethod
    def zeros(cls, dim: int = None):
        """Create a zero vertex in 2D or 3D space.

        Parameters
        ----------
        dim : int, optional
            The dimension of the vertex (2 or 3). Defaults to 3.

        Returns
        -------
        Vertex
            The zero vertex.
        """
        dim = dim if not None else 3
        return cls(torch.zeros(dim, dtype=DTYPE, device=DEVICE))
    
    def plot(self, fig=None, color='black', marker='circle', size=5, label=None):
        """Plot the point using  [Plotly](https://plotly.com/).

        Parameters
        ----------
        fig : plotly.graph_objects.Figure, optional
            The Plotly figure to add the point to. If None, a new figure will be created.
        color : str, optional
            The color of the point. Defaults to 'black'.
        marker : str, optional
            The marker style. Defaults to 'circle'.
        size : int, optional
            The size of the marker. Defaults to 5.
        label : str, optional
            The label to display for the point. If None, no label will be displayed.

        Returns
        -------
        plotly.graph_objects.Figure
            The  [Plotly](https://plotly.com/) figure object.

        Raises
        ------
        ImportError
            If  [Plotly](https://plotly.com/) is not installed.
        ValueError
            If the point is not 2D or 3D.
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            raise ImportError("Plotly is required for plotting.")
        
        if self.dim not in [2, 3]:
            raise ValueError("Plotting is only supported for 2D and 3D points.")
        
        x, y = self.data[0].cpu().numpy(), self.data[1].cpu().numpy()
        
        if self.dim == 2:
            trace = go.Scatter(
                x=[x], y=[y], mode='markers', marker=dict(color=color, symbol=marker, size=size),
                text=[label] if label else None, hoverinfo='text' if label else 'none'
            )
        else:  # self.dim == 3
            z = self.data[2].cpu().numpy()
            trace = go.Scatter3d(
                x=[x], y=[y], z=[z], mode='markers', marker=dict(color=color, symbol=marker, size=size),
                text=[label] if label else None, hoverinfo='text' if label else 'none'
            )
        
        if fig is None:
            fig = go.Figure(data=[trace])
            return fig
        else:
            fig.add_trace(trace)
            return fig


class Direction(VectorBase):
    """
    A direction vector in 2D or 3D.

    In geometry, direction is the same as spatial direction or vector direction. It is the common characteristic of
    components that can be made equal by scaling. Therefore, a direction is represented here as a _unit vector_ similar
    to an axis.
    
    Attributes
    ----------
    components : torch.Tensor
        The direction components.
    dim : int
        The dimension of the rotation matrix (2 or 3).
    system : str, optional
        The coordinate system of the vector. Allowed values are 'cartesian', 'cylindrical',
        'spherical', or 'polar'. Default is 'cartesian'.
    """
    
    def __init__(self, components=None, dim: int = None, system: str = 'cartesian'):
        """
        Initialize a [`Direction`][pymagbot.math.spatial.primitives.Direction].

        Parameters
        ----------
        components : array_like, optional
            The coordinates of the direction vector.
        dim : int, optional
            The dimension of the direction vector (2 or 3).
        """
        super().__init__(components, dim=dim, system=system)
        self.normalize()
    
    def normalize(self):
        """Normalize the vector to unit length."""
        with torch.no_grad():
            self._data = self._data / torch.linalg.norm(self.data)
    
    @MatrixBase.data.setter
    def data(self, value):
        """Override setter to enforce normalization."""
        MatrixBase.data.fset(self, value)
        self.normalize()
    
    def plot(self, fig=None, origin=None, color='red', scale=1.0, label=None):
        """Plot the direction vector using  [Plotly](https://plotly.com/).

        Parameters
        ----------
        fig: plotly.graph_objects.Figure, optional
            The Plotly figure to add the direction vector to. If None, a new figure will be created.
        origin : Point, optional
            The origin point of the direction vector. If None, the origin is assumed to be (0, 0, 0).
        color : str, optional
            The color of the direction vector. Defaults to 'r' (red).
        scale : float, optional
            The scale of the direction vector. Defaults to 1.0.
        label : str, optional
            The label to display for the direction vector. If None, no label will be displayed.

        Returns
        -------
        plotly.graph_objects.Figure
            The  [Plotly](https://plotly.com/) figure object.

        Raises
        ------
        ImportError
            If  [Plotly](https://plotly.com/) is not installed.
        ValueError
            If the direction vector is not 2D or 3D.
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            raise ImportError("Plotly is required for plotting. Install with `pip install plotly`.")
        
        if self.dim not in [2, 3]:
            raise ValueError("Plotting is only supported for 2D and 3D direction components.")
        
        if origin is None:
            origin = Vertex(torch.zeros(self.dim), dim=self.dim)
        
        x0, y0 = origin.data[0].cpu().numpy(), origin.data[1].cpu().numpy()
        x1 = x0 + self.data[0].cpu().numpy() * scale
        y1 = y0 + self.data[1].cpu().numpy() * scale
        
        if self.dim == 2:
            trace = go.Scatter(
                x=[x0, x1], y=[y0, y1], mode='lines', line=dict(color=color), text=[label] if label else None,
                hoverinfo='text' if label else 'none'
            )
        else:  # self.dim == 3
            z0, z1 = origin.data[2].cpu().numpy(), origin.data[2].cpu().numpy() + self.data[2].cpu().numpy() * scale
            trace = go.Scatter3d(
                x=[x0, x1], y=[y0, y1], z=[z0, z1], mode='lines', line=dict(color=color),
                text=[label] if label else None, hoverinfo='text' if label else 'none'
            )
        
        if fig is None:
            fig = go.Figure(data=[trace])
            return fig
        else:
            fig.add_trace(trace)
            return fig
