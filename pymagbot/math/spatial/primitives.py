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

from pymagbot.math.matrix_base import ensure_tensor, MatrixBase
from pymagbot.utils.config import DEVICE, DTYPE

__all__ = ['VectorBase', 'Vertex', 'Direction']


class VectorBase(MatrixBase):
    """Base class for vector-like primitives.
    
    A vector is a tuple of $n$ (here $n=2$ or 3) real numbers has a natural structure of vector space defined by
    component-wise addition and scalar multiplication.
    This class also provides common operations for vectors, including coordinate system conversions.
    
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
    _allowed_systems = {'cartesian', 'cylindrical', 'spherical', 'polar'}
    _systems: str = 'cartesian'
    
    def __init__(self, components: Optional[Union[torch.Tensor, ArrayLike, 'VectorBase', list, tuple]] = None,
                 dim: int = None, system: str = 'cartesian'):
        super().__init__(components, dim=dim)
        self.system = system
    
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
            data = components.clone()
        else:
            data = torch.as_tensor(components, dtype=DTYPE, device=self.device)
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
    
    @property
    def system(self) -> str:
        """Get the coordinate system of the vector."""
        return self._system
    
    @system.setter
    def system(self, new_system: str):
        new_system = new_system.lower()
        if new_system not in self._allowed_systems:
            raise ValueError(f"system must be one of {self._allowed_systems}, got {new_system}")
        self._system = new_system
    
    def __repr__(self) -> str:
        """String representation of the object."""
        return f"{self.__class__.__name__}(components={self.array.tolist()})"
    
    @property
    def components(self) -> torch.Tensor:
        """The [`VectorBase`][pymagbot.math.spatial.primitives.VectorBase] components. """
        return self._data.squeeze()
    
    @components.setter
    def components(self, value):
        """Setter for the components."""
        self.data = value
    
    @property
    def x(self):
        """The first element in the [`VectorBase`][pymagbot.math.spatial.primitives.VectorBase]'s data."""
        return self.components[0]
    
    @property
    def y(self):
        """The second element in the [`VectorBase`][pymagbot.math.spatial.primitives.VectorBase]'s data."""
        return self.components[1]
    
    @property
    def z(self):
        """The third element in the  [`VectorBase`][pymagbot.math.spatial.primitives.VectorBase]'s data."""
        if self.dim < 3:
            raise (IndexError(f"z Coordinates accessible in 3D!"))
        return self.components[2]
    
    @property
    def magnitude(self) -> torch.Tensor:
        """Get the magnitude  (i.e. Euclidean norm) of the vector."""
        return torch.linalg.vector_norm(self.components, ord=2)
    
    @magnitude.setter
    def magnitude(self, mag):
        """The vector magnitude (Euclidean norm)"""
        self.scale(mag / self.magnitude)
    
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
    
    def to_polar(self) -> 'VectorBase':
        """
        Convert a 2D vector from Cartesian to polar coordinates.

        Returns
        -------
        VectorBase
            A new instance with polar coordinates (r, phi), where r is the magnitude and phi is the angle.
        """
        # Assume self.data is [x, y]
        if self.dim != 2:
            raise ValueError("to_polar() only supports 2D vectors.")
        x, y = self.to_list()
        r = torch.sqrt(x ** 2 + y ** 2)
        phi = torch.atan2(y, x)
        return self.__class__([r, phi], dim=2, system='polar')
    
    def to_cylindrical(self) -> 'VectorBase':
        """Convert the vector from Cartesian to cylindrical coordinates.

        For a 3D vector, the cylindrical coordinates are defined as:
            (r, phi, z)
        where:
            - r is the radial component,
            - phi is the azimuthal angle (in radians),
            - z is the vertical component.

        For a 2D vector, z is omitted.

        Returns
        -------
        VectorBase
            The cylindrical components (r, phi, z) for 3D, or (r, phi, 0) for 2D.
        """
        
        if self.system in ['cylindrical', 'polar']:
            return self.clone()
        if self.dim == 2:
            return self.to_polar()
        elif self.dim == 3:
            r, phi = self.to_polar().to_list()
            return VectorBase((r, phi, self.z), dim=self.dim, system='cylindrical')
        else:
            raise ValueError("to_cylindrical() only supports 2D or 3D vectors.")
    
    def to_spherical(self) -> 'VectorBase':
        """
        Convert the vector from Cartesian to spherical coordinates.

        For a 2D vector, spherical conversion is equivalent to polar representation.
        For a 3D vector, returns (r, theta, phi).

        Returns
        -------
        VectorBase
            A new instance with the vector in spherical (or polar for 2D) coordinates.
        """
        if self.dim == 2:
            return self.to_polar()
        elif self.dim == 3:
            r, phi = self.to_polar().to_list()
            theta = 0.0 if r == 0 else torch.acos(self.z / r)
            return self.__class__([r, theta, phi], dim=3, system='spherical')
        else:
            raise ValueError("to_spherical() only supports 2D and 3D vectors.")
        
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
        return {"components": self.array.tolist()}
    
    @classmethod
    def from_dict(cls, data):
        return cls(data["components"])


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
    
    def __init__(self, components=None, dim: int = None, system: str= 'cartesian'):
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
