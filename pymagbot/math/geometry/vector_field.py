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
""" [VectorField][pymagbot.math.geometry.VectorField] - Vector field class.

A vector field is a function that assigns a vector to every point in a subset of space.
In this module, we define the [`VectorField`][pymagbot.math.geometry.VectorField] class, which represents a vector field
in 2D or 3D space. The vector field is defined on a grid of points, and the vector components are stored in a
[`Mesh`][pymagbot.math.geometry.Mesh] object. The class provides methods for computing the gradient, divergence,
curl, and interpolating the vector field at arbitrary points. It also allows for expressing the vector field in
different reference frames and converting the object to and from a dictionary representation.
"""

from typing import List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import ArrayLike

from pymagbot.math.geometry.geometry_base import GeometryBase
from pymagbot.math.geometry.mesh import Mesh
from pymagbot.math.geometry.span import Span
from pymagbot.math.spatial.frame import Frame
from pymagbot.utils.config import DEVICE, DTYPE


class VectorField(GeometryBase):
    """Represents a vector field in 2D or 3D space.

    A vector field is an assignment of a vector to each point in space.  Internally,
    the [`VectorField`][pymagbot.math.geometry.VectorField] stores the vector components at each grid point of a
    [`Mesh`][pymagbot.math.geometry.Mesh].

    Attributes
    ----------
    components : torch.Tensor
        The vector components at each point in the mesh.
    points : Mesh
        The mesh grid defining the points in space where the components are defined.
    dim: int
        The base dimension (2 or 3) of the geometric object.
    frame : Frame
        The reference frame of the geometry object. Default the [world frame][pymagbot.math.frame.Frame.world] is
        used.
    name : str
        The unique name of the geometry object.
        
    Examples:
    --------
    ```python
    import torch
    from pymagbot.math.frame import Frame
    from pymagbot.math.geometry import VectorField, Mesh, Span

    # Define the grid
    mesh = Mesh(Span(0, 1, 10))
    
    # Define the vector components
    vf_x_components = mesh.y * torch.exp(mesh.x)
    vf_y_components = mesh.x**2 + torch.exp(mesh.x)
    vf_z_components = mesh.z**2 * torch.exp(mesh.z)
    vector_field = VectorField(components = (vf_x_components, vf_z_components, vf_z_components),
                               points=mesh, frame=Frame.world)
    print(vector_field)
    ```

    """
    
    # components = super().data  # Ensure that the vector components should be stored in _data attribute from MatrixBase
    
    # Special Methods
    def __init__(self, components: Union[ArrayLike, torch.Tensor, list, tuple], points: Mesh = None,
                 dim: int = None, frame: Frame = None, name: Optional[str] = None):
        """Initialize a VectorField.

        Parameters
        ----------
        components : torch.Tensor
            Vector components at each point in the mesh.
        points : Mesh, optional
            The mesh grid defining the points in space where the vectors are defined.
        dim : int, optional
            The mesh dimension (2 or 3).
        frame : Frame, optional
            The reference frame of the mesh.
        name : str, optional
            The unique name of the vector field.
        """
        components = self._initialize_data(components)
        if points is None:
            dim = dim if dim is not None else components.shape[-1]  # infer the dimension from the shape of vector
            spans: List[Span] = []
            for k in range(dim - 1):
                # Set the default values for new spans dimension to set the new default mesh
                span = Span(start=0, stop=1, num_samples=components.shape[k])
                spans.append(span)
                points = Mesh(spans=spans, dim=dim)
        else:
            dim = dim if dim is not None else points.dim
        if not isinstance(points, Mesh):
            raise TypeError("points must be a Mesh compatible object.")
        self.mesh: Mesh = points
        super().__init__(data=components, dim=dim, frame=frame, name=name)
    
    def __repr__(self) -> str:
        """String representation of the object."""
        return f"VectorField(name='{self.name}', components.shape={self.components.shape}, mesh={self.mesh}," \
               f" frame='{self.frame.name}')"
    
    def express_in_frame(self, frame: Frame) -> 'VectorField':
        """Express this vector field in a different frame.
        
        This method does not change the underlying geometry but returns a *new*
        vector field object whose properties (e.g., coordinates, vectors) are
        expressed with respect to the `frame`.

        Parameters
        ----------
        frame : Frame
            The frame in which to express the geometry.
        
        Returns
        -------
        VectorField
            A *new* VectorField object with its properties expressed in the `frame`.
        """
        if self.mesh.dim != frame.dim:
            raise ValueError(
                f"Mesh dimension ({self.mesh.dim}) and RigidTransform dimension ({frame.dim}) must match.")
        
        new_mesh = self.mesh.express_in_frame(frame)
        # Apply rotation to each vector in the field
        original_shape = self.components.shape
        reshaped_vectors = self.components.reshape(-1, self.mesh.dim).T  # (dim, N)
        rotated_vectors = frame.local_rotation.data @ reshaped_vectors  # (dim, N)
        components = rotated_vectors.T.reshape(original_shape)
        return VectorField(components=components, points=new_mesh, frame=frame, name=f"{self.name}_in_{frame.name}")
    
    def gradient(self, method: str = 'fd', padding_mode: str = 'zeros') -> 'VectorField':
        """Compute the gradient of the scalar field.

        Parameters
        ----------
        method : str, optional
            The method used to compute the gradient. Options are `fd` (finite difference).
        padding_mode : str, optional
            The padding mode used for the finite difference method (see [`torch.nn.functional.pad`][torch.nn.functional.pad]). Default: `zeros`

        Returns
        -------
        VectorField
            A new VectorField object representing the gradient.
        """
        self._check_method(method)
        
        # Use F.pad to pad the components before calculating gradients
        vx, vy = self._get_pad(padding_mode)
        # if self.dim == 3:
        #     vz = F.pad(self.vz.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode=padding_mode).squeeze()
        
        # Calculate the gradient using finite difference
        grad_x = torch.gradient(vx, spacing=self.mesh.spans[0].step, dim=0)[0]  # First order derivative
        grad_y = torch.gradient(vy, spacing=self.mesh.spans[1].step, dim=1)[0]  # First order derivative
        
        if self.dim == 2:
            gradient = torch.stack([grad_x[1:-1, 1:-1], grad_y[1:-1, 1:-1]], dim=-1)  # Crop the edges after calculation
        elif self.dim == 3:
            vz = F.pad(self.vz.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode=padding_mode).squeeze()
            grad_z = torch.gradient(vz, spacing=self.mesh.spans[2].step, dim=2)[0]  # First order derivative
            gradient = torch.stack([grad_x[1:-1, 1:-1, 1:-1], grad_y[1:-1, 1:-1, 1:-1], grad_z[1:-1, 1:-1, 1:-1]],
                                   dim=-1)  # Crop the edges after calculation
        else:
            raise ValueError("Gradient is only supported for 2D and 3D vector fields.")
        
        return VectorField(components=gradient, points=self.points, frame=self.frame, name=f"{self.name}_gradient")
    
    def divergence(self, method: str = 'fd', padding_mode: str = 'zeros') -> 'torch.Tensor':
        """Compute the divergence of the vector field.

        Parameters
        ----------
        method : str, optional
            The method used to compute the divergence. Options are `fd` (finite difference).
        padding_mode : str, optional
            The padding mode used for the finite difference method
            see [`torch.nn.functional.pad`][torch.nn.functional.pad]). Default: `zeros`

        Returns
        -------
        torch.Tensor
            A new scalar field representing the divergence.
        """
        self._check_method(method)
        
        # Use F.pad to pad the components before calculating gradients
        vx, vy = self._get_pad(padding_mode)
        # Calculate the derivatives using finite difference
        dvx_dx = torch.gradient(vx, spacing=self.mesh.spans[0].step, dim=0)[0]
        dvy_dy = torch.gradient(vy, spacing=self.mesh.spans[1].step, dim=1)[0]
        if self.dim == 2:
            divergence = dvx_dx[1:-1, 1:-1] + dvy_dy[1:-1, 1:-1]  # Crop the edges after calculation
        elif self.dim == 3:
            vz = F.pad(self.vz.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode=padding_mode).squeeze()
            dvz_dz = torch.gradient(vz, spacing=self.mesh.spans[2].step, dim=2)[0]
            divergence = dvx_dx[1:-1, 1:-1, 1:-1] + dvy_dy[1:-1, 1:-1, 1:-1] + dvz_dz[1:-1, 1:-1, 1:-1]
        else:
            raise ValueError("Divergence is only supported for 2D and 3D vector fields.")
        
        return divergence
    
    def curl(self, method: str = 'fd', padding_mode: str = 'zeros') -> Union['VectorField', torch.Tensor]:
        """Compute the curl of the vector field.

        Parameters
        ----------
        method : str, optional
            The method used to compute the curl. Options are `fd` (finite difference).
        padding_mode : str, optional
            The padding mode used for the finite difference method
            (see [`torch.nn.functional.pad`][torch.nn.functional.pad]).

        Returns
        -------
        Union[VectorField, torch.Tensor]
            A new VectorField object representing the curl (in 3D) or a scalar field (in 2D).
        """
        self._check_method(method)
        
        # Use F.pad to pad the components before calculating derivatives
        vx = F.pad(self.vx.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode=padding_mode).squeeze()
        vy = F.pad(self.vy.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode=padding_mode).squeeze()
        
        # Calculate the derivatives using finite difference
        dvx_dy = torch.gradient(vx, spacing=self.mesh.spans[1].step, dim=1)[0]
        dvy_dx = torch.gradient(vy, spacing=self.mesh.spans[0].step, dim=0)[0]
        
        if self.dim == 2:
            curl = dvy_dx[1:-1, 1:-1] - dvx_dy[1:-1, 1:-1]  # Crop the edges after calculation
            return curl
        elif self.dim == 3:
            vz = F.pad(self.vz.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode=padding_mode).squeeze()
            dvx_dz = torch.gradient(vx, spacing=self.mesh.spans[2].step, dim=2)[0]
            dvy_dz = torch.gradient(vy, spacing=self.mesh.spans[2].step, dim=2)[0]
            dvz_dx = torch.gradient(vz, spacing=self.mesh.spans[0].step, dim=0)[0]
            dvz_dy = torch.gradient(vz, spacing=self.mesh.spans[1].step, dim=1)[0]
            
            curl_x = dvy_dz[1:-1, 1:-1, 1:-1] - dvz_dy[1:-1, 1:-1, 1:-1]  # Crop the edges after calculation
            curl_y = dvz_dx[1:-1, 1:-1, 1:-1] - dvx_dz[1:-1, 1:-1, 1:-1]  # Crop the edges after calculation
            curl_z = dvy_dx[1:-1, 1:-1, 1:-1] - dvx_dy[1:-1, 1:-1, 1:-1]  # Crop the edges after calculation
            
            curl = torch.stack([curl_x, curl_y, curl_z], dim=-1)
            return VectorField(components=curl, points=self.points, frame=self.frame, name=f"{self.name}_curl")
        else:
            raise ValueError("Curl is only supported for 2D and 3D vector fields.")
    
    def interpolate(self, points: Union[ArrayLike, torch.Tensor, 'Mesh', Sequence[Span], list, tuple],
                    method: str = 'linear', padding_mode: str = 'zeros', align_corners: bool = False) -> 'VectorField':
        """Interpolate vector components at specified points.
        
        This method interpolates the vector field at the given points using `torch.nn.functional.grid_sample`.
        The input points are normalized to the range of [-1, 1] based on the spans of the mesh grid.
        The grid_sample function then performs the interpolation using the specified method and padding mode.

        Parameters
        ----------
        points : Union[ArrayLike, torch.Tensor, Mesh, Sequence[Span], list, tuple]
            Points at which to interpolate. Can be:
                - An array-like, torch.Tensor, list, or tuple: points at which to interpolate, shape=(N, dim).
                - A Mesh object: The points will be the point into the grid
                - A Sequence[Span]: spans the new mesh point
        method : str, optional
            Interpolation method. Options are 'linear'. Default is `linear`.
        padding_mode : str, optional
            The padding mode used for the finite difference method
            (see [`torch.nn.functional.grid_sample`][torch.nn.functional.grid_sample]).
            Default is `zeros`
        align_corners : bool, optional
            Parameter passed to [`torch.nn.functional.grid_sample`][torch.nn.functional.grid_sample]. Default: `False`

        Returns
        -------
        torch.Tensor
            Interpolated vector components, shape=(N, dim).
        """
        if method != 'linear':
            raise NotImplementedError("Only linear interpolation is currently supported.")
        
        if isinstance(points, (ArrayLike, torch.Tensor, list, tuple)):
            points_tensor = torch.as_tensor(points, dtype=DTYPE, device=DEVICE)
        elif isinstance(points, Mesh):
            if points.dim != self.dim:
                raise ValueError(f"Points must have the same dim as the vectorfield ({self.dim})")
            points_tensor = points.grid.view(-1, self.dim)  # Reshape the data to 2D
        elif isinstance(points, Sequence):
            # points are a sequence of Span, must recreate the point and transform it
            interpolation_mesh = Mesh(points)
            points_tensor = interpolation_mesh.grid.view(-1, self.dim)  # Reshape the data to 2D
        else:
            raise TypeError("Points must be a list, tuple, np.ndarray, torch.Tensor, or Mesh object.")
        
        if points_tensor.shape[-1] != self.dim:
            raise ValueError(f"The dimension of interpolation points must match vector field dimension ({self.dim}).")
        
        # Normalize point coordinates to [-1, 1] range for grid_sample
        normalized_points = self._normalize_points(points_tensor)
        # grid_sample expects input in the format (N, C, H, W) for 2D or (N, C, D, H, W) for 3D
        # where N is the batch size, C is the number of channels (components), and H, W, D are the dimensions
        # of the grid. We set N=1 and C=dim.
        if self.dim == 2:
            components = self.components.permute(2, 0, 1).unsqueeze(0)  # (N, C, H, W)
            normalized_points = normalized_points.view(1, 1, -1, self.dim)
        elif self.dim == 3:
            components = self.components.permute(3, 0, 1, 2).unsqueeze(0)  # (N, C, D, H, W)
            normalized_points = normalized_points.view(1, 1, -1, self.dim)
        else:
            raise ValueError("VectorField dimension must be 2 or 3.")
        interpolated_components = F.grid_sample(
            components,
            normalized_points,
            mode=method,
            padding_mode=padding_mode,
            align_corners=align_corners
        )
        interpolated_components = interpolated_components.squeeze().permute(1, 0)
        interpolation_spans = Span.from_tensor(points_tensor)
        interpolation_points = Mesh(spans=interpolation_spans, dim=self.dim)
        # Create a new VectorField with the interpolated components
        return VectorField(components=interpolated_components, points=interpolation_points, frame=self.frame,
                           name=f"{self.name}_interpolated")
    
    def _initialize_data(self, components) -> torch.Tensor:
        """Check the initial `components` to be sure of their shape and type.
        If not valid, raise an exception.
        """
        if not isinstance(components, torch.Tensor):
            if isinstance(components, (list, tuple)):
                if len(components) in (2, 3):
                    components = torch.stack(components, dim=-1)
                else:
                    raise ValueError(f"VectorField initialized with list or tuple should be of dim 2 or 3,"
                                     " got {len(components)}!")
            elif isinstance(components, np.ndarray):
                components = torch.as_tensor(components, dtype=DTYPE, device=DEVICE)
            else:
                raise TypeError("components must be a list, tuple, np.ndarray or torch.Tensor")
        return components
    
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
            dict_obj['components'] = self.components.tolist()
            dict_obj['points'] = self.mesh.to_dict(full)
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
        components = torch.as_tensor(dict_obj['components'], dtype=DTYPE, device=DEVICE)
        points = Mesh.from_dict(dict_obj['points'], known)
        return cls(components=components, points=points, name=name, frame=frame)
    
    @property
    def components(self) -> torch.Tensor:
        """Get the vector data"""
        return self.data
    
    @components.setter
    def components(self, components: torch.Tensor):
        """Set the  vector  data"""
        self.data = components
        self._validate()
    
    @property
    def vx(self) -> torch.Tensor:
        """Get the x-components of the vector field."""
        if self.dim < 1:
            raise ValueError("Mesh dimension must be at least 1.")
        return self.components[..., 0]
    
    @vx.setter
    def vx(self, value: torch.Tensor):
        """Set the x-components of the vector field."""
        if self.dim < 1:
            raise ValueError("Mesh dimension must be at least 1.")
        expected_shape = self.components.shape[:-1]
        if value.shape != expected_shape:
            raise ValueError(f"X-component shape {value.shape} does not match expected shape {expected_shape}")
        self.components[..., 0] = value
    
    @property
    def vy(self) -> torch.Tensor:
        """Get the y-components of the vector field."""
        if self.dim < 2:
            raise ValueError("Mesh dimension must be at least 2.")
        return self.components[..., 1]
    
    @vy.setter
    def vy(self, value: torch.Tensor):
        """Set the y-components of the vector field."""
        if self.dim < 2:
            raise ValueError("Mesh dimension must be at least 2.")
        expected_shape = self.components.shape[:-1]
        if value.shape != expected_shape:
            raise ValueError(f"Y-component shape {value.shape} does not match expected shape {expected_shape}")
        self.components[..., 1] = value
    
    @property
    def vz(self) -> torch.Tensor:
        """Get the z-components of the vector field."""
        if self.dim < 3:
            raise ValueError("Mesh dimension must be at least 3.")
        return self.components[..., 2]
    
    @vz.setter
    def vz(self, value: torch.Tensor):
        """Set the z-components of the vector field."""
        if self.dim < 3:
            raise ValueError("Mesh dimension must be at least 3.")
        expected_shape = self.components.shape[:-1]
        if value.shape != expected_shape:
            raise ValueError(f"Z-component shape {value.shape} does not match expected shape {expected_shape}")
        self.components[..., 2] = value
    
    @property
    def points(self) -> Mesh:
        """Get the mesh grid points. """
        return self.mesh
    
    def _validate(self):
        """Validates that the `components` tensor has the correct shape.

        Raises
        ------
        ValueError
            If the shape of `components` is incompatible with the shape of the `mesh`.
        """
        if self.mesh.dim != self.dim:
            raise ValueError("Points must have the same dim as the dimension in its parameters' space/mesh")
        expected_shape = (*self.mesh.grid.shape[:-1], self.dim)
        if self.components.shape != expected_shape:
            raise ValueError(f"The shape of 'components' must be {expected_shape}, but got {self.components.shape}")
    
    @staticmethod
    def _check_method(method: str):
        """Check if the method is supported."""
        if method not in ['fd']:
            raise ValueError(f"Method '{method}' is not supported."
                             f"Currently supported methods are: 'fd'.")
    
    def _get_pad(self, padding_mode: str = 'zeros') -> tuple:
        # Use F.pad to pad the components before calculating gradients
        vx = F.pad(self.vx.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode=padding_mode).squeeze()
        vy = F.pad(self.vy.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode=padding_mode).squeeze()
        return vx, vy
    
    # def _get_derivative(self, padding_mode: str = 'zeros') -> tuple
    #     vx, vy = self._get_pad(padding_mode)
    #     # Calculate the derivatives using finite difference
    #     dvx_dx = torch.gradient(vx, spacing=self.mesh.spans[0].step, dim=0)[0]
    #     dvy_dy = torch.gradient(vy, spacing=self.mesh.spans[1].step, dim=1)[0]
    #
    #     return vx, vy, dvx_dx, dvy_dy
    
    def _normalize_points(self, points: torch.Tensor) -> torch.Tensor:
        """Normalize points to the range of [-1, 1] based on the spans of the mesh.

        Parameters
        ----------
        points : torch.Tensor
            Points to normalize, shape: (N, dim).

        Returns
        -------
        torch.Tensor
            Normalized points, shape: (N, dim).
        """
        normalized_points = torch.empty_like(points, dtype=DTYPE, device=DEVICE)
        for i in range(self.dim):
            normalized_points[:, i] = self.mesh.spans[i].normalize(points[:, i])
        
        return normalized_points
