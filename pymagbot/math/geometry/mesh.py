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
""" The `mesh` module."""

from typing import Optional, Sequence, Union

import torch
from numpy.typing import ArrayLike

from pymagbot.math.spatial.frame import Frame
from pymagbot.math.geometry.geometry_base import GeometryBase
from pymagbot.math.geometry.span import Span
from pymagbot.utils.config import DEVICE, DTYPE


class Mesh(GeometryBase):
    """Represents a mesh grid of Cartesian coordinates in 2D or 3D space.

    A mesh is a division of an interval (or [`Span`][pymagbot.math.geometry.Span]) that is a finite sequence of
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
        The reference frame of the geometry object. Default the [world frame][pymagbot.math.spatial.frame.Frame.world] is
        used.
    name : str
        The unique name of the geometry object.
    indexing: str
        The indexing mode, either “xy” or “ij”, defaults to “ij” (see also [torch.meshgrid][]).
    x, y, z : torch.Tensor
        The grid coordinates in the x, y, and z directions.
    
    Examples
    --------
    ```python
    import torch
    from pymagbot.math.frame import Frame
    from pymagbot.math.geometry import Mesh, Span
    
    #Define the mesh grid in 3D with 3 different span
    span_x = Span(0, 1, 10)
    span_y = Span(0, 1, 10)
    span_z = Span(0, 1, 10)
    mesh1 = Mesh([span_x, span_y, span_z], frame=Frame.world)
    
    # Define a mesh grid from a single span, dim=2 force 2D, the same span is used on every axes
    mesh2 = Mesh(Span(0, 1, 10), dim=2, frame=Frame.world)
    
    # Define a mesh grid from a single span, as default 3D, the same span is used on every axes
    mesh3 = Mesh(Span(0, 1, 10), frame=Frame.world)
    ```
    """
    
    # Class Attributes
    grid_x: Optional[torch.Tensor] = None
    """The grid coordinates in the x direction."""
    grid_y: Optional[torch.Tensor] = None
    """The grid coordinates in the y direction."""
    grid_z: Optional[torch.Tensor] = None
    """The grid coordinates in the z direction."""
    
    # Special Methods
    def __init__(self, spans: Union[Sequence[Span], Span], dim: int = None, frame: Frame = None,
                 name: Optional[str] = None, indexing: str = 'ij'):
        """Initialize a [`Mesh`][pymagbot.math.geometry.Mesh].

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
            The indexing mode, either "y" or "ij", defaults to "ij" (see also [torch.meshgrid][]).
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
            raise ValueError("The dim must be the same of the length of spans")
        super().__init__(data=self._create_grid(self.indexing), dim=self.dim, frame=frame, name=name)
    
    def __repr__(self) -> str:
        """String representation of the [`Mesh`][pymagbot.math.geometry.Mesh]."""
        return f"Mesh(name='{self.name}', spans={self.spans}, shape={self.grid.shape}, frame='{self.frame.name}')"
        
    # Public Instance Methods
    def express_in_frame(self, frame: Frame) -> 'Mesh':
        """Express this [`Mesh`][pymagbot.math.geometry.Mesh] in a different [`frame`][pymagbot.math.Frame].

        This method does not change the underlying geometry but returns a *new*
        geometry object whose properties (e.g., coordinates, components) are
        expressed with respect to the `target_frame`.

        Parameters
        ----------
        frame : Frame
            The frame in which to express the geometry.
        Returns
        -------
        Mesh
            A *new* [`Mesh`][pymagbot.math.geometry.Mesh] object with its properties expressed in the `frame`.
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
    
    def resize(self, spans: Sequence[Span]):
        """Resize the mesh grid with new spans.

        Parameters
        ----------
        spans : Sequence[Span]
            The new spans.
        """
        self.spans = spans
        
    def interpolate(self, points: Union[ArrayLike, torch.Tensor, 'Mesh', Sequence[Span], list, tuple],
                    method: str = 'linear', padding_mode: str = 'zeros', align_corners: bool = False) -> 'Mesh':
        """Interpolate coordinates at specified points.
    
        Parameters
        ----------
        points : Union[ArrayLike, torch.Tensor, Mesh, list, tuple]
            Points at which to interpolate, shape=(N, dim) or a Mesh object.
        method : str, optional
            Interpolation method. Only 'linear' is currently supported. Default is `linear`.
        padding_mode : str, optional
            The padding mode used for the finite difference method
            (see [`torch.nn.functional.grid_sample`][torch.nn.functional.grid_sample]).
            Default is `zeros`.
        align_corners : bool, optional
            Parameter passed to [`torch.nn.functional.grid_sample`][torch.nn.functional.grid_sample]. Default is `False`.
    
        Returns
        -------
        Mesh
            Interpolated coordinates, shape=(N, dim).
        Raises
        ------
        NotImplementedError
            If the interpolation method is not 'linear'.
        TypeError
            If the points are not of the correct type.
        ValueError
            If the points have the wrong dimension.

        References
        ----------
        - Pytorch documentation  [`torch.nn.functional.grid_sample`][torch.nn.functional.grid_sample]
        """
        if method != 'linear':
            raise NotImplementedError("Only linear interpolation is currently supported.")
        
        if isinstance(points, (ArrayLike, torch.Tensor, list, tuple)):
            points_tensor = torch.as_tensor(points, dtype=DTYPE, device=DEVICE)
        elif isinstance(points, Mesh):
            if points.dim != self.dim:
                raise ValueError(f"Points must have the same dim as the origin ({self.dim})")
            points_tensor = points.grid.reshape(-1, self.dim)
        elif isinstance(points, Sequence):
            points = Mesh(spans=points, dim=self.dim, frame=self.frame)
            points_tensor = points.grid.reshape(-1, self.dim)
        else:
            raise TypeError("Points must be a list, tuple, np.ndarray, torch.Tensor, or Mesh object.")
        # Normalize the points to the range of [-1, 1] based on the spans
        normalized_points = self._normalize_points(points_tensor)
        # Reshape the normalized points to (1, 1, N, dim) for grid_sample
        normalized_points = normalized_points.reshape(1, 1, -1, self.dim)
        # Transpose the grid to (1, dim, H, W) for 2D or (1, dim, H, W, D) for 3D
        if self.dim == 2:
            grid = self.grid.permute(2, 0, 1).unsqueeze(0)  # (1, dim, H, W)
        elif self.dim == 3:
            grid = self.grid.permute(3, 0, 1, 2).unsqueeze(0)  # (1, dim, H, W, D)
        else:
            raise ValueError("Mesh dimension must be 2 or 3.")

        # Use grid_sample to interpolate the grid at the normalized points
        interpolated_values = torch.nn.functional.grid_sample(grid, normalized_points, mode=method,
                                                              padding_mode=padding_mode, align_corners=align_corners)
        # Reshape the interpolated values to (N, dim)
        interpolated_values = interpolated_values.squeeze().T
        # Create new spans from the interpolated values
        interpolated_spans = Span.from_tensor(interpolated_values)
        return Mesh(spans=interpolated_spans, frame=self.frame, name=f"{self.name}_interpolated")
    
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
        """Create a [`Mesh`][pymagbot.math.geometry.Mesh] from a [`pandas.DataFrame`][pandas.DataFrame]."""
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
    
    @property
    def grid(self) -> torch.Tensor:
        """Get the mesh grid points, shape: $(..., \\text{dim})$."""
        return self.data
    
    @grid.setter
    def grid(self, value):
        """Set new values to the grid points"""
        self.data = value
        self._create_coordinates()
    
    points = property(grid)
    """Alias to the grid property."""
    
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
        
    def _validate(self):
        """Validate the shape and properties of the [`Mesh`][pymagbot.math.geometry.Mesh]."""
        if len(self.spans) != self.dim:
            raise ValueError(f"Number of spans ({len(self.spans)}) must match the mesh dimension ({self.dim}).")
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
    
    def _create_coordinates(self):
        """Create coordinates to easy manage the points.

        Check the dimension of the grid points
        """
        if self.dim == 2:
            self.grid_x, self.grid_y = self.grid.split([1, 1], dim=-1)
        elif self.dim == 3:
            self.grid_x, self.grid_y, self.grid_z = self.grid.split([1, 1, 1], dim=-1)

    def _normalize_points(self, points: torch.Tensor) -> torch.Tensor:
        """Normalize points to the range of [-1, 1] based on the spans.

        Parameters
        ----------
        points : torch.Tensor
            Points to normalize, shape: (N, dim).

        Returns
        -------
        torch.Tensor
            Normalized points, shape: (N, dim).
        """
        normalized_points = torch.empty_like(points)
        for i in range(self.dim):
            normalized_points[:, i] = self._spans[i].normalize(points[:, i])
        return normalized_points
    