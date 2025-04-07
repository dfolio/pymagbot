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
from platform import system
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import ArrayLike
from pandas.core.dtypes.cast import infer_dtype_from

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
    system : str
        The coordinate system of the vector field.
        
    Examples
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
    # Attributes
    _gradient: Union['VectorField', None] = None
    _divergence: Union[torch.Tensor, None] = None
    _curl: Union['VectorField', torch.Tensor, None] = None

    _allowed_systems = {'cartesian', 'cylindrical', 'spherical', 'polar'}
    _system: str = 'cartesian'
    
    # Special Methods
    def __init__(self, components: Union[ArrayLike, torch.Tensor, list, tuple],
                 points: Union[Mesh, Span, Sequence[Span]] = None,
                 dim: int = None, frame: Frame = None, name: Optional[str] = None, system: Optional[str] = 'cartesian'):
        """Initialize a VectorField.

        Parameters
        ----------
        components : array_like
            Vector field components at each point in the mesh.
        points : Mesh, Span, optional
            The mesh grid defining the points in space where the vectors are defined.
        dim : int, optional
            The mesh dimension (2 or 3).
        frame : Frame, optional
            The reference frame of the mesh.
        name : str, optional
            The unique name of the vector field.
        system : str, optional
            The coordinate system of the vector field. Defaults to 'cartesian'.
        """
        if system not in self._allowed_systems:
            raise ValueError(f"Invalid coordinate system: {system}. Allowed systems are: {self._allowed_systems}")
        components = self._initialize_data(components)
        inferred_dim = components.shape[-1]
        if points is None:
            dim = dim if dim is not None else inferred_dim  # infer the dimension from the shape of vector
            spans: List[Span] = []
            component_len = components.shape[0] if components.ndim > 1 else 1
            for k in range(dim - 1):
                # Set the default values for new spans dimension to set the new default mesh
                span = Span(start=0, stop=1, num_samples=component_len)
                spans.append(span)
            points = Mesh(spans=spans, dim=dim, frame=frame)
        elif isinstance(points, (Span, Sequence[Span])):
            points = Mesh(spans=points, dim=dim, frame=frame)
        else:
            if not isinstance(points, Mesh):
                raise TypeError("points must be a Mesh compatible object.")
            dim = dim if dim is not None else points.dim
            if dim != inferred_dim:
                raise ValueError(f"The dimension of the mesh ({points.dim}) must match the"
                                 f"number of components in the vector field ({inferred_dim}).")
        self.mesh: Mesh = points
        self.system: str = system
        super().__init__(data=components, dim=dim, frame=frame, name=name)
        self._invalidate_cache()
    
    def __repr__(self) -> str:
        """String representation of the object."""
        return f"VectorField(name='{self.name}', components.shape={self.components.shape}, mesh={self.mesh}," \
               f" frame='{self.frame.name}', system='{self.system}'')"
    
    def to_cartesian(self) -> 'VectorField':
        """Convert the vector field to Cartesian coordinates.

        Returns
        -------
        VectorField
            A new VectorField object with components in Cartesian coordinates.
        """
        if self.system == 'cartesian':
            return self
        return self._coordinate_transform('cartesian')

    def to_cylindrical(self) -> 'VectorField':
        """Convert the vector field to cylindrical coordinates.

        Returns
        -------
        VectorField
            A new VectorField object with components in cylindrical coordinates.
        """
        if self.system == 'cylindrical':
            return self
        return self._coordinate_transform('cylindrical')

    def to_spherical(self) -> 'VectorField':
        """Convert the vector field to spherical coordinates.

        Returns
        -------
        VectorField
            A new VectorField object with components in spherical coordinates.
        """
        if self.system == 'spherical':
            return self
        return self._coordinate_transform('spherical')

    def to_polar(self) -> 'VectorField':
        """Convert the vector field to polar coordinates (only for 2D vector fields).

        Returns
        -------
        VectorField
            A new VectorField object with components in polar coordinates.
        """
        if self.system == 'polar':
            return self
        return self._coordinate_transform('polar')
    
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
        if self.dim != frame.dim:
            raise ValueError(f"VectorField dimension ({self.dim}) and Frame dimension ({frame.dim}) must match.")
        new_mesh = self.mesh.express_in_frame(frame)
        # Apply rotation to each vector in the field
        original_shape = self.components.shape
        reshaped_vectors = self.components.reshape(-1, self.mesh.dim).T  # (dim, N)
        rotated_vectors = frame.local_rotation.data @ reshaped_vectors  # (dim, N)
        components = rotated_vectors.T.reshape(original_shape)
        return VectorField(components=components, points=new_mesh, frame=frame, name=f"{self.name}_in_{frame.name}")
    
    # def gradient(self, method: str = 'fd', padding_mode: str = 'zeros') -> 'VectorField':
    #     """Compute the gradient of the scalar field.
    #
    #     Parameters
    #     ----------
    #     method : str, optional
    #         The method used to compute the gradient. Options are `fd` (finite difference).
    #     padding_mode : str, optional
    #         The padding mode used for the finite difference method (see [`torch.nn.functional.pad`][torch.nn.functional.pad]). Default: `zeros`
    #
    #     Returns
    #     -------
    #     VectorField
    #         A new VectorField object representing the gradient.
    #     """
    #     self._check_method(method)
    #
    #     # Use F.pad to pad the components before calculating gradients
    #     vx, vy = self._get_pad(padding_mode)
    #     # if self.dim == 3:
    #     #     vz = F.pad(self.vz.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode=padding_mode).squeeze()
    #
    #     # Calculate the gradient using finite difference
    #     grad_x = torch.gradient(vx, spacing=self.mesh.spans[0].step, dim=0)[0]  # First order derivative
    #     grad_y = torch.gradient(vy, spacing=self.mesh.spans[1].step, dim=1)[0]  # First order derivative
    #
    #     if self.dim == 2:
    #         gradient = torch.stack([grad_x[1:-1, 1:-1], grad_y[1:-1, 1:-1]], dim=-1)  # Crop the edges after calculation
    #     elif self.dim == 3:
    #         vz = F.pad(self.vz.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode=padding_mode).squeeze()
    #         grad_z = torch.gradient(vz, spacing=self.mesh.spans[2].step, dim=2)[0]  # First order derivative
    #         gradient = torch.stack([grad_x[1:-1, 1:-1, 1:-1], grad_y[1:-1, 1:-1, 1:-1], grad_z[1:-1, 1:-1, 1:-1]],
    #                                dim=-1)  # Crop the edges after calculation
    #     else:
    #         raise ValueError("Gradient is only supported for 2D and 3D vector fields.")
    #
    #     return VectorField(components=gradient, points=self.points, frame=self.frame, name=f"{self.name}_gradient")
    #
    # def divergence(self, method: str = 'fd', padding_mode: str = 'zeros') -> 'torch.Tensor':
    #     """Compute the divergence of the vector field.
    #
    #     Parameters
    #     ----------
    #     method : str, optional
    #         The method used to compute the divergence. Options are `fd` (finite difference).
    #     padding_mode : str, optional
    #         The padding mode used for the finite difference method
    #         see [`torch.nn.functional.pad`][torch.nn.functional.pad]). Default: `zeros`
    #
    #     Returns
    #     -------
    #     torch.Tensor
    #         A new scalar field representing the divergence.
    #     """
    #     self._check_method(method)
    #
    #     # Use F.pad to pad the components before calculating gradients
    #     vx, vy = self._get_pad(padding_mode)
    #     # Calculate the derivatives using finite difference
    #     dvx_dx = torch.gradient(vx, spacing=self.mesh.spans[0].step, dim=0)[0]
    #     dvy_dy = torch.gradient(vy, spacing=self.mesh.spans[1].step, dim=1)[0]
    #     if self.dim == 2:
    #         divergence = dvx_dx[1:-1, 1:-1] + dvy_dy[1:-1, 1:-1]  # Crop the edges after calculation
    #     elif self.dim == 3:
    #         vz = F.pad(self.vz.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode=padding_mode).squeeze()
    #         dvz_dz = torch.gradient(vz, spacing=self.mesh.spans[2].step, dim=2)[0]
    #         divergence = dvx_dx[1:-1, 1:-1, 1:-1] + dvy_dy[1:-1, 1:-1, 1:-1] + dvz_dz[1:-1, 1:-1, 1:-1]
    #     else:
    #         raise ValueError("Divergence is only supported for 2D and 3D vector fields.")
    #
    #     return divergence
    #
    # def curl(self, method: str = 'fd', padding_mode: str = 'zeros') -> Union['VectorField', torch.Tensor]:
    #     """Compute the curl of the vector field.
    #
    #     Parameters
    #     ----------
    #     method : str, optional
    #         The method used to compute the curl. Options are `fd` (finite difference).
    #     padding_mode : str, optional
    #         The padding mode used for the finite difference method
    #         (see [`torch.nn.functional.pad`][torch.nn.functional.pad]).
    #
    #     Returns
    #     -------
    #     Union[VectorField, torch.Tensor]
    #         A new VectorField object representing the curl (in 3D) or a scalar field (in 2D).
    #     """
    #     self._check_method(method)
    #
    #     # Use F.pad to pad the components before calculating derivatives
    #     vx = F.pad(self.vx.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode=padding_mode).squeeze()
    #     vy = F.pad(self.vy.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode=padding_mode).squeeze()
    #
    #     # Calculate the derivatives using finite difference
    #     dvx_dy = torch.gradient(vx, spacing=self.mesh.spans[1].step, dim=1)[0]
    #     dvy_dx = torch.gradient(vy, spacing=self.mesh.spans[0].step, dim=0)[0]
    #
    #     if self.dim == 2:
    #         curl = dvy_dx[1:-1, 1:-1] - dvx_dy[1:-1, 1:-1]  # Crop the edges after calculation
    #         return curl
    #     elif self.dim == 3:
    #         vz = F.pad(self.vz.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode=padding_mode).squeeze()
    #         dvx_dz = torch.gradient(vx, spacing=self.mesh.spans[2].step, dim=2)[0]
    #         dvy_dz = torch.gradient(vy, spacing=self.mesh.spans[2].step, dim=2)[0]
    #         dvz_dx = torch.gradient(vz, spacing=self.mesh.spans[0].step, dim=0)[0]
    #         dvz_dy = torch.gradient(vz, spacing=self.mesh.spans[1].step, dim=1)[0]
    #
    #         curl_x = dvy_dz[1:-1, 1:-1, 1:-1] - dvz_dy[1:-1, 1:-1, 1:-1]  # Crop the edges after calculation
    #         curl_y = dvz_dx[1:-1, 1:-1, 1:-1] - dvx_dz[1:-1, 1:-1, 1:-1]  # Crop the edges after calculation
    #         curl_z = dvy_dx[1:-1, 1:-1, 1:-1] - dvx_dy[1:-1, 1:-1, 1:-1]  # Crop the edges after calculation
    #
    #         curl = torch.stack([curl_x, curl_y, curl_z], dim=-1)
    #         return VectorField(components=curl, points=self.points, frame=self.frame, name=f"{self.name}_curl")
    #     else:
    #         raise ValueError("Curl is only supported for 2D and 3D vector fields.")
    
    def interpolate(self, points: Union[ArrayLike, torch.Tensor, 'Mesh', Sequence[Span], Span, list, tuple],
                    method: str = 'linear', padding_mode: str = 'zeros', align_corners: bool = False) -> 'VectorField':
        """Interpolate vector components at specified points.
        
        This method interpolates the vector field at the given points using
        [`torch.nn.functional.grid_sample`][torch.nn.functional.grid_sample].
        The input points are normalized to the range of [-1, 1] based on the spans of the mesh grid.
        The grid_sample function then performs the interpolation using the specified method and padding mode.

        Parameters
        ----------
        points : array_like
            Points at which to interpolate. Can be:
                - An array-like, torch.Tensor, list, or tuple: points at which to interpolate, shape=(N, dim).
                - A Mesh object: The points will be the point into the grid
                - A Sequence[Span], Span: spans the new mesh point
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
        .. [1] Pytorch documentation, [`torch.nn.functional.grid_sample`][torch.nn.functional.grid_sample]
        """
        if method != 'linear':
            raise NotImplementedError("Only linear interpolation is currently supported.")
        
        if isinstance(points, (ArrayLike, torch.Tensor, list, tuple)):
            points_tensor = torch.as_tensor(points, dtype=DTYPE, device=DEVICE)
        elif isinstance(points, Mesh):
            if points.dim != self.dim:
                raise ValueError(f"Points (dim={points.dim}) must have the same dim as the VectorField ({self.dim})")
            points_tensor = points.grid.view(-1, self.dim)  # Reshape the data to 2D
        elif isinstance(points, (Sequence, Span)):
            # points are a sequence of Span, must recreate the point and transform it
            interpolation_mesh = Mesh(points, dim=self.dim, frame=self.frame)
            points_tensor = interpolation_mesh.grid.view(-1, self.dim)  # Reshape the data to 2D
        else:
            raise TypeError("Points must be a list, tuple, np.ndarray, torch.Tensor, or Mesh object.")
        
        if points_tensor.shape[-1] != self.dim:
            raise ValueError(f"The dimension of interpolation points  (dim={points.dim}) must match "
                             f"vector field dimension ({self.dim}).")
        
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
        interpolated_components = F.grid_sample(components, normalized_points,
                                                mode=method, padding_mode=padding_mode, align_corners=align_corners)
        interpolated_components = interpolated_components.squeeze().permute(1, 0)
        interpolation_spans = Span.from_tensor(points_tensor)
        interpolation_points = Mesh(spans=interpolation_spans, dim=self.dim, frame=self.frame)
        # Create a new VectorField with the interpolated components
        return VectorField(components=interpolated_components, points=interpolation_points, frame=self.frame,
                           name=f"{self.name}_interpolated")
    
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
        dict_obj['system'] = self.system
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
        system = dict_obj.get('system', 'cartesian')
        components_data = dict_obj.get('components', [])
        components = torch.as_tensor(components_data, dtype=DTYPE, device=DEVICE)
        
        # Retrieve 'points' from dict_obj; default to an empty dict if not present
        points_data = dict_obj.get('points', {})
        points = Mesh.from_dict(points_data, known) if len(components_data) > 0 else None
        return cls(components=components, points=points, name=name, frame=frame, system=system)
    
    @property
    def system(self) -> str:
        """Get the coordinate system of the vector field."""
        return self._system

    @system.setter
    def system(self, value: str):
        """Set the coordinate system of the vector field."""
        if value not in self._allowed_systems:
            raise ValueError(f"Invalid coordinate system: {value}. Allowed systems are: {self._allowed_systems}")
        self._system = value
        
    @property
    def components(self) -> torch.Tensor:
        """Get the vector data"""
        return self.data
    
    @components.setter
    def components(self, components: torch.Tensor):
        """Set the  vector  data"""
        self.data = components
        self._validate()
        self._invalidate_cache()  # Reset cached values when components change
    
    @property
    def vx(self) -> torch.Tensor:
        """Get the x-components of the vector field."""
        if self.system != 'cartesian' or self.dim < 1:
            raise AttributeError("x-component is only valid for Cartesian 2D/3D VectorFields.")
        return self.components[..., 0]
    
    @vx.setter
    def vx(self, value: torch.Tensor):
        """Set the x-components of the vector field."""
        if self.system != 'cartesian' or self.dim < 1:
            raise AttributeError("x-component is only valid for Cartesian 2D/3D VectorFields.")
        self._set_component(0, value)
    
    @property
    def vy(self) -> torch.Tensor:
        """Get the y-component of the vector field (Cartesian)."""
        if self.system != 'cartesian' or self.dim < 2:
            raise AttributeError("y-component is only valid for Cartesian 2D/3D VectorFields.")
        return self.components[..., 1]
    
    @vy.setter
    def vy(self, value: torch.Tensor):
        """Set the y-component of the vector field (Cartesian)."""
        if self.system != 'cartesian' or self.dim < 2:
            raise AttributeError("y-component is only valid for Cartesian 2D/3D VectorFields.")
        self._set_component(1, value)
    
    @property
    def vz(self) -> torch.Tensor:
        """Get the z-component of the vector field (Cartesian)."""
        if self.system != 'cartesian' or self.dim < 3:
            raise AttributeError("z-component is only valid for Cartesian 3D VectorFields.")
        return self.components[..., 2]
    
    @vz.setter
    def vz(self, value: torch.Tensor):
        """Set the z-component of the vector field (Cartesian)."""
        if self.system != 'cartesian' or self.dim < 3:
            raise AttributeError("z-component is only valid for Cartesian 3D VectorFields.")
        self._set_component(2, value)
    
    @property
    def vr(self) -> torch.Tensor:
        """Get the radial component of the vector field (Spherical/Cylindrical/Polar)."""
        if self.system not in ('spherical', 'cylindrical', 'polar') or self.dim < 1:
            raise AttributeError("Radial component is only valid for Spherical/Cylindrical/Polar VectorFields.")
        return self.components[..., 0]
    
    @vr.setter
    def vr(self, value: torch.Tensor):
        """Set the radial component of the vector field (Spherical/Cylindrical/Polar)."""
        if self.system not in ('spherical', 'cylindrical', 'polar') or self.dim < 1:
            raise AttributeError("Radial component is only valid for Spherical/Cylindrical/Polar VectorFields.")
        self._set_component(0, value)
    
    @property
    def v_theta(self) -> torch.Tensor:
        """Get the azimuthal angle component of the vector field (Spherical/Polar)."""
        if self.system not in ('spherical', 'polar') or self.dim < 2:
            raise AttributeError("Azimuthal angle component is only valid for Spherical/Polar VectorFields.")
        return self.components[..., 1]
    
    @v_theta.setter
    def v_theta(self, value: torch.Tensor):
        """Set the azimuthal angle component of the vector field (Spherical/Polar)."""
        if self.system not in ('spherical', 'polar') or self.dim < 2:
            raise AttributeError("Azimuthal angle component is only valid for Spherical/Polar VectorFields.")
        self._set_component(1, value)
    
    @property
    def v_phi(self) -> torch.Tensor:
        """Get the polar angle component of the vector field (Spherical/Cylindrical)."""
        if self.system not in ('spherical', 'cylindrical') or self.dim < 2:
            raise AttributeError("Polar angle component is only valid for Spherical/Cylindrical VectorFields.")
        return self.components[..., 1] if self.system == 'cylindrical' else self.components[..., 2]
    
    @v_phi.setter
    def v_phi(self, value: torch.Tensor):
        """Set the polar angle component of the vector field (Spherical/Cylindrical)."""
        if self.system not in ('spherical', 'cylindrical') or self.dim < 2:
            raise AttributeError("Polar angle component is only valid for Spherical/Cylindrical VectorFields.")
        self._set_component(1 if self.system == 'cylindrical' else 2, value)
    
    @property
    def v_rho(self) -> torch.Tensor:
        """Get the rho component of the vector field (Cylindrical)."""
        return self.vr
    
    @v_rho.setter
    def v_rho(self, value: torch.Tensor):
        """Set the rho component of the vector field (Cylindrical)."""
        self.vr = value

    @property
    def points(self) -> Mesh:
        """Get the mesh grid points. """
        return self.mesh

    @property
    def gradient(self) -> 'VectorField':
        """Compute and cache the gradient of the vector field."""
        if self._gradient is None:
            self._gradient = self._compute_gradient()
        return self._gradient

    @property
    def divergence(self) -> torch.Tensor:
        """Compute and cache the divergence of the vector field."""
        if self._divergence is None:
            self._divergence = self._compute_divergence()
        return self._divergence

    @property
    def curl(self) -> Union['VectorField', torch.Tensor]:
        """Compute and cache the curl of the vector field."""
        if self._curl is None:
            self._curl = self._compute_curl()
        return self._curl

    # Private Methods
    def _initialize_data(self, components: Union[ArrayLike, torch.Tensor, list, tuple]) -> torch.Tensor:
        """Check the initial `components` to be sure of their shape and type.
         
         This method handles various input types for the vector field components,
         converting them into a consistent `torch.Tensor` format. It supports
        `torch.Tensor`, `numpy.ndarray`, lists, and tuples of tensors or arrays.

        Parameters
        ----------
        components : array_like
            Vector components that can be a single tensor or array, or a list/tuple
            of tensors or arrays representing the components.

        Returns
        -------
        torch.Tensor
            A tensor containing the vector field components.

        Raises
        ------
        TypeError
            If the input type is not supported.
        ValueError
            If the input has an invalid shape or dimension.
        """

        if isinstance(components, torch.Tensor):
            return components.to(DTYPE).to(DEVICE)
        if isinstance(components, np.ndarray):
            return torch.as_tensor(components, dtype=DTYPE, device=DEVICE)
        if isinstance(components, (list, tuple)):
            # Check if the list/tuple contains tensors or arrays
            if len(components) not in (2, 3):
                raise ValueError(f"VectorField initialized with list or tuple should be of dim 2 or 3, "
                                 f"got {len(components)}!")
            tensor_list = []
            for component in components:
                if isinstance(component, torch.Tensor):
                    tensor_list.append(component.to(DTYPE).to(DEVICE))
                elif isinstance(component, np.ndarray):
                    tensor_list.append(torch.as_tensor(component, dtype=DTYPE, device=DEVICE))
                else:
                    raise TypeError("Each component in the list/tuple must be a torch.Tensor or a np.ndarray")

            # Stack the tensors along the last dimension
            try:
                components = torch.stack(tensor_list, dim=-1)
            except RuntimeError as e:
                raise ValueError("The components could not be stacked. Ensure they have compatible shapes.") from e
            return components

        raise TypeError("components must be a torch.Tensor, np.ndarray, list, or tuple of tensors/arrays")
        
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
   
    # --- Caching mechanism for computed derivatives --- #
    def _invalidate_cache(self):
        """Invalidate cached computed properties."""
        self._gradient = None
        self._divergence = None
        self._curl = None
        
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
        normalized_points = torch.empty_like(points, dtype=DTYPE, device=self.device)
        for i in range(self.dim):
            normalized_points[:, i] = self.mesh.spans[i].normalize(points[:, i])
        return normalized_points

    def _pad_components(self, padding_mode: str = 'zeros') -> List[torch.Tensor]:
        """Pad the vector field components.

        Parameters
        ----------
        padding_mode : str, optional
            The padding mode used for [`torch.nn.functional.pad`][torch.nn.functional.pad]. Default is `zeros`.

        Returns
        -------
        List[torch.Tensor]
            Padded components.
        """
        # Use F.pad to pad the components before calculating derivatives
        vx = F.pad(self.vx.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode=padding_mode).squeeze()
        vy = F.pad(self.vy.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode=padding_mode).squeeze()
        vz = F.pad(self.vz.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode=padding_mode).squeeze() if self.dim == 3 else None
        return [vx, vy, vz]
    
    def _compute_derivatives(self, padded_components: List[torch.Tensor]) -> List[torch.Tensor]:
        """Compute the derivatives of the vector field components.

        Parameters
        ----------
        padded_components : List[torch.Tensor]
            List of padded components derivative.

        Returns
        -------
        List[torch.Tensor]
            List of derivative tensors.
        """
        vx, vy, vz = padded_components
        dvx_dx = torch.gradient(vx, spacing= self.mesh.spans[0].step, dim=0)[0]
        dvy_dy = torch.gradient(vy, spacing= self.mesh.spans[1].step, dim=1)[0]

        if self.dim == 3:
            step_z = self.mesh.spans[2].step
            dvz_dz = torch.gradient(vz, spacing=step_z, dim=2)[0]
            return [dvx_dx, dvy_dy, dvz_dz]
        return [dvx_dx, dvy_dy]

    def _compute_gradient(self, method: str = 'fd', padding_mode: str = 'zeros') -> 'VectorField':
        """Compute the gradient of the vector field.

        Parameters
        ----------
        method : str, optional
            The method used to compute the gradient. Options are `fd` (finite difference). Default is `fd`.
        padding_mode : str, optional
            The padding mode used for the finite difference method (see [`torch.nn.functional.pad`][torch.nn.functional.pad]).
            Default is `zeros`.

        Returns
        -------
        VectorField
            A new VectorField object representing the gradient.
        """
        if method != 'fd':
            raise NotImplementedError("Only finite difference method is currently supported.")

        padded_components = self._pad_components(padding_mode)
        grad_x = torch.gradient(padded_components[0], spacing=self.mesh.spans[0].step, dim=0)[0]
        grad_y = torch.gradient(padded_components[1], spacing=self.mesh.spans[1].step, dim=1)[0]

        if self.dim == 2:
            gradient = torch.stack([grad_x[1:-1, 1:-1], grad_y[1:-1, 1:-1]], dim=-1)
        elif self.dim == 3:
            grad_z = torch.gradient(padded_components[2], spacing=self.mesh.spans[2].step, dim=2)[0]
            gradient = torch.stack([grad_x[1:-1, 1:-1, 1:-1], grad_y[1:-1, 1:-1, 1:-1], grad_z[1:-1, 1:-1, 1:-1]], dim=-1)
        else:
            raise ValueError("Gradient is only supported for 2D and 3D vector fields.")

        return VectorField(components=gradient, points=self.points, frame=self.frame, name=f"{self.name}_gradient")
    
    def _compute_divergence(self, method: str = 'fd', padding_mode: str = 'zeros') -> torch.Tensor:
        """Compute the divergence of the vector field.

        Parameters
        ----------
        method : str, optional
            The method used to compute the divergence. Options are `fd` (finite difference). Default is `fd`.
        padding_mode : str, optional
            The padding mode used for the finite difference method
            (see [`torch.nn.functional.pad`][torch.nn.functional.pad]). Default is `zeros`.

        Returns
        -------
        torch.Tensor
            A new scalar field representing the divergence.
        """
        if method != 'fd':
            raise NotImplementedError("Only finite difference method is currently supported.")

        padded_components = self._pad_components(padding_mode)
        derivatives = self._compute_derivatives(padded_components[:2])

        if self.dim == 2:
            divergence = derivatives[0][1:-1, 1:-1] + derivatives[1][1:-1, 1:-1]
        elif self.dim == 3:
            divergence = derivatives[0][1:-1, 1:-1, 1:-1] + derivatives[1][1:-1, 1:-1, 1:-1] + \
                         self._compute_derivatives(padded_components)[2][1:-1, 1:-1, 1:-1]
        else:
            raise ValueError("Divergence is only supported for 2D and 3D vector fields.")

        return divergence

    def _compute_curl(self, method: str = 'fd', padding_mode: str = 'zeros') -> Union['VectorField', torch.Tensor]:
        """Compute the curl of the vector field.

        Parameters
        ----------
        method : str, optional
            The method used to compute the curl. Options are `fd` (finite difference). Default is `fd`.
        padding_mode : str, optional
            The padding mode used for the finite difference method
            (see [`torch.nn.functional.pad`][torch.nn.functional.pad]). Default is `zeros`.

        Returns
        -------
        Union[VectorField, torch.Tensor]
            A new VectorField object representing the curl (in 3D) or a scalar field (in 2D).
        """
        if method != 'fd':
            raise NotImplementedError("Only finite difference method is currently supported.")

        padded_components = self._pad_components(padding_mode)
        vx, vy, vz = padded_components[0], padded_components[1], padded_components[2] if len(padded_components) == 3 else None

        step_x = self.mesh.spans[0].step
        step_y = self.mesh.spans[1].step

        dvx_dy = torch.gradient(vx, spacing=step_y, dim=1)[0]
        dvy_dx = torch.gradient(vy, spacing=step_x, dim=0)[0]

        if self.dim == 2:
            curl = dvy_dx[1:-1, 1:-1] - dvx_dy[1:-1, 1:-1]
            return curl
        elif self.dim == 3:
            step_z = self.mesh.spans[2].step
            dvx_dz = torch.gradient(vx, spacing=step_z, dim=2)[0]
            dvy_dz = torch.gradient(vy, spacing=step_z, dim=2)[0]
            dvz_dx = torch.gradient(vz, spacing=step_x, dim=0)[0]
            dvz_dy = torch.gradient(vz, spacing=step_y, dim=1)[0]

            curl_x = dvy_dz[1:-1, 1:-1, 1:-1] - dvz_dy[1:-1, 1:-1, 1:-1]
            curl_y = dvz_dx[1:-1, 1:-1, 1:-1] - dvx_dz[1:-1, 1:-1, 1:-1]
            curl_z = dvy_dx[1:-1, 1:-1, 1:-1] - dvx_dy[1:-1, 1:-1, 1:-1]

            curl = torch.stack([curl_x, curl_y, curl_z], dim=-1)
            return VectorField(components=curl, points=self.points, frame=self.frame, name=f"{self.name}_curl")
        else:
            raise ValueError("Curl is only supported for 2D and 3D vector fields.")

    def _coordinate_transform(self, target_system: str) -> 'VectorField':
        """General coordinate transformation method.

        Parameters
        ----------
        target_system : str
            The target coordinate system.

        Returns
        -------
        VectorField
            A new VectorField object in the target coordinate system.
        """
        if target_system not in self._allowed_systems:
            raise ValueError(f"Invalid target coordinate system: {target_system}")

        if self.dim == 2:
            if self.system == 'cartesian' and target_system == 'polar':
                x = self.points.x
                y = self.points.y
                vx = self.components[..., 0]
                vy = self.components[..., 1]

                r = torch.sqrt(x ** 2 + y ** 2)
                theta = torch.atan2(y, x)

                vr = vx * torch.cos(theta) + vy * torch.sin(theta)
                vtheta = -vx * torch.sin(theta) + vy * torch.cos(theta)

                components = torch.stack([vr, vtheta], dim=-1)
                new_mesh = Mesh(spans=(Span(min(r), max(r), self.mesh.spans[0].num),
                                      Span(min(theta), max(theta), self.mesh.spans[1].num)), dim=2)
                return VectorField(components=components, points=new_mesh, frame=self.frame, system='polar',
                                   name=f"{self.name}_to_polar")
            elif self.system == 'polar' and target_system == 'cartesian':
                r = self.points.x
                theta = self.points.y
                vr = self.components[..., 0]
                vtheta = self.components[..., 1]

                vx = vr * torch.cos(theta) - vtheta * torch.sin(theta)
                vy = vr * torch.sin(theta) + vtheta * torch.cos(theta)

                components = torch.stack([vx, vy], dim=-1)
                return VectorField(components=components, points=self.points, frame=self.frame, system='cartesian',
                                   name=f"{self.name}_to_cartesian")
            else:
                raise NotImplementedError(f"Conversion from {self.system} to {target_system} is not supported.")
        elif self.dim == 3:
            if self.system == 'cartesian' and target_system == 'cylindrical':
                x = self.points.x
                y = self.points.y
                z = self.points.z
                vx = self.components[..., 0]
                vy = self.components[..., 1]
                vz = self.components[..., 2]

                rho = torch.sqrt(x ** 2 + y ** 2)
                phi = torch.atan2(y, x)

                vrho = vx * torch.cos(phi) + vy * torch.sin(phi)
                vphi = -vx * torch.sin(phi) + vy * torch.cos(phi)
                vz = vz

                components = torch.stack([vrho, vphi, vz], dim=-1)
                new_mesh = Mesh(spans=(Span(min(rho), max(rho), self.mesh.spans[0].num),
                                      Span(min(phi), max(phi), self.mesh.spans[1].num),
                                      Span(min(z), max(z), self.mesh.spans[2].num)), dim=3)
                return VectorField(components=components, points=new_mesh, frame=self.frame, system='cylindrical',
                                   name=f"{self.name}_to_cylindrical")
            elif self.system == 'cylindrical' and target_system == 'cartesian':
                rho = self.points.x
                phi = self.points.y
                z = self.points.z
                vrho = self.components[..., 0]
                vphi = self.components[..., 1]
                vz = self.components[..., 2]

                vx = vrho * torch.cos(phi) - vphi * torch.sin(phi)
                vy = vrho * torch.sin(phi) + vphi * torch.cos(phi)
                vz = vz

                components = torch.stack([vx, vy, vz], dim=-1)
                return VectorField(components=components, points=self.points, frame=self.frame, system='cartesian',
                                   name=f"{self.name}_to_cartesian")
            elif self.system == 'cartesian' and target_system == 'spherical':
                x = self.points.x
                y = self.points.y
                z = self.points.z
                vx = self.components[..., 0]
                vy = self.components[..., 1]
                vz = self.components[..., 2]

                r = torch.sqrt(x ** 2 + y ** 2 + z ** 2)
                theta = torch.acos(z / r)
                phi = torch.atan2(y, x)

                vr = vx * torch.sin(theta) * torch.cos(phi) + \
                     vy * torch.sin(theta) * torch.sin(phi) + \
                     vz * torch.cos(theta)
                vtheta = vx * torch.cos(theta) * torch.cos(phi) + \
                         vy * torch.cos(theta) * torch.sin(phi) - \
                         vz * torch.sin(theta)
                vphi = -vx * torch.sin(phi) + vy * torch.cos(phi)

                components = torch.stack([vr, vtheta, vphi], dim=-1)
                new_mesh = Mesh(spans=(Span(min(r), max(r), self.mesh.spans[0].num),
                                      Span(min(theta), max(theta), self.mesh.spans[1].num),
                                      Span(min(phi), max(phi), self.mesh.spans[2].num)), dim=3)
                return VectorField(components=components, points=new_mesh, frame=self.frame, system='spherical',
                                   name=f"{self.name}_to_spherical")
            elif self.system == 'spherical' and target_system == 'cartesian':
                x = self.points.x
                y = self.points.y
                z = self.points.z
                vx = self.components[..., 0]
                vy = self.components[..., 1]
                vz = self.components[..., 2]

                r = torch.sqrt(x ** 2 + y ** 2 + z ** 2)
                theta = torch.acos(z / r)
                phi = torch.atan2(y, x)

                vr = vx * torch.sin(theta) * torch.cos(phi) + \
                     vy * torch.sin(theta) * torch.sin(phi) + \
                     vz * torch.cos(theta)
                vtheta = vx * torch.cos(theta) * torch.cos(phi) + \
                         vy * torch.cos(theta) * torch.sin(phi) - \
                         vz * torch.sin(theta)
                vphi = -vx * torch.sin(phi) + vy * torch.cos(phi)

                components = torch.stack([vr, vtheta, vphi], dim=-1)
                return VectorField(components=components, points=self.points, frame=self.frame, system='cartesian',
                                   name=f"{self.name}_to_cartesian")
            elif self.system == 'cylindrical' and target_system == 'spherical':
                rho = self.points.x
                phi = self.points.y
                z = self.points.z
                vrho = self.components[..., 0]
                vphi = self.components[..., 1]
                vz = self.components[..., 2]

                r = torch.sqrt(rho ** 2 + z ** 2)
                theta = torch.atan2(rho, z)

                vr = vrho * torch.sin(theta) + vz * torch.cos(theta)
                vtheta = vrho * torch.cos(theta) - vz * torch.sin(theta)
                vphi = vphi

                components = torch.stack([vr, vtheta, vphi], dim=-1)
                new_mesh = Mesh(spans=(Span(min(r), max(r), self.mesh.spans[0].num),
                                      Span(min(theta), max(theta), self.mesh.spans[1].num),
                                      Span(min(phi), max(phi), self.mesh.spans[2].num)), dim=3)
                return VectorField(components=components, points=new_mesh, frame=self.frame, system='spherical',
                                   name=f"{self.name}_to_spherical")
            elif self.system == 'spherical' and target_system == 'cylindrical':
                r = self.points.x
                theta = self.points.y
                phi = self.points.z
                vr = self.components[..., 0]
                vtheta = self.components[..., 1]
                vphi = self.components[..., 2]

                vrho = vr * torch.sin(theta) + vtheta * torch.cos(theta)
                vphi = vphi
                vz = vr * torch.cos(theta) - vtheta * torch.sin(theta)

                components = torch.stack([vrho, vphi, vz], dim=-1)
                return VectorField(components=components, points=self.points, frame=self.frame, system='cylindrical',
                                   name=f"{self.name}_to_cylindrical")
            else:
                raise NotImplementedError(f"Conversion from {self.system} to {target_system} is not supported.")
        else:
            raise ValueError("Coordinate transformation is only supported for 2D and 3D vector fields.")

    def _set_component(self, index: int, value: torch.Tensor):
        """Set a component of the vector field and reset derived quantities."""
        expected_shape = self.components.shape[:-1]
        if value.shape != expected_shape:
            raise ValueError(f"Component shape {value.shape} does not match expected shape {expected_shape}")
        new_components = self.components.clone()  # Create a copy to avoid modifying the original tensor in place
        new_components[..., index] = value
        self.components = new_components
        self._invalidate_cache()