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


from typing import Optional, Union

import torch
from numpy.typing import ArrayLike

from pymagbot._config import USE_CUDA
from pymagbot.math.matrix_base import MatrixBase, VectorBase

__all__ = ['RotationMatrix', 'Vertex', 'Direction', 'RigidTransform']


class RotationMatrix(MatrixBase):
    """A  rotation matrix representing an element of  SO(2) or SO(3).
    
    In linear algebra, a rotation matrix is a transformation matrix that is used to perform a rotation in Euclidean
    space. More specifically, they can be characterized as **orthogonal matrice**, that is, a _square matrix_ $R$ is a
    rotation matrix if and only if
    $R^T = R^{−1}$ and $\\mathrm{det}(R) = 1$.
    
    The set of all orthogonal matrices of size $n$ is a representation of a group known as the
    _special orthogonal group_ $SO(n)$ (here $n=$2 or 3),
    Every rotation maps an orthonormal basis of $\\mathbb{R}^n$ to another orthonormal basis.
    """
    
    def __init__(self, matrix: Optional[Union[torch.Tensor, ArrayLike, 'RotationMatrix', list, tuple]] = None,
                 dim: int = 3):
        """
        Initialize the [`RotationMatrix`][pymagbot.math.geometry_base.RotationMatrix].

        Parameters
        ----------
        matrix : array_like, optional
            The rotation matrix data.
        dim : int, optional
            The dimension of the rotation matrix (2 or 3).
        """
        super().__init__(matrix if matrix is not None else torch.eye(dim), dim=dim)
    
    def _validate(self):
        """
        Ensure that the matrix is a valid rotation matrix (orthogonal and determinant=1).

        Raises
        ------
        ValueError
            If the matrix is not orthogonal or the determinant is not close to 1.
        """
        if self.data.shape != (self.dim, self.dim):
            raise ValueError(f"Rotation matrix must be square with dimension {self.dim}, got {self.data.shape}.")
        if not torch.allclose(self.data.T @ self.data, torch.eye(self.dim, device=self.device), atol=1e-6):
            raise ValueError(f"Rotation matrix must be orthogonal, got {self.data}.")
        if not torch.isclose(torch.det(self.data), torch.tensor(1.0, device=self.device), atol=1e-6):
            raise ValueError(f"Rotation matrix determinant must be 1, got {torch.det(self.data)}")
    
    @classmethod
    def check(cls, rotation: Union[torch.Tensor, ArrayLike, 'RotationMatrix', list, tuple]) -> 'RotationMatrix':
        """Check if rotation is or can be converted to [`RotationMatrix`][pymagbot.math.geometry_base.RotationMatrix]

        Parameters
        ----------
        rotation : Union[torch.Tensor, ArrayLike, RotationMatrix, list, tuple]
            The rotation to check or convert.

        Returns
        -------
        RotationMatrix
            The converted  [`RotationMatrix`][pymagbot.math.geometry_base.RotationMatrix].

        Raises
        ------
        TypeError
            If the input cannot be converted to a RotationMatrix.
        """
        if isinstance(rotation, (torch.Tensor, ArrayLike, list, tuple)):
            rotation = cls(rotation)
        if not isinstance(rotation, RotationMatrix):
            raise TypeError('other is not a valid rotation matrix.')
        return rotation
    
    def inverse(self) -> 'RotationMatrix':
        """ Return the inverse of the rotation matrix.

        Returns
        -------
        RotationMatrix
            The inverse rotation matrix.
        """
        return RotationMatrix(self.data.T, dim=self.dim)
    
    inv = inverse
    
    def trace(self) -> torch.Tensor:
        """Return the trace of the [`RotationMatrix`][pymagbot.math.geometry_base.RotationMatrix].

        Returns
        -------
        torch.Tensor
            The trace of the matrix.
        """
        return self.data.trace()
    
    def angle(self):
        """Returns absolute deviation of the  [`RotationMatrix`][pymagbot.math.geometry_base.RotationMatrix]
         from identity

        Returns
        -------
        torch.Tensor
            The rotation angle in radians.
        """
        cos_theta = (self.trace() - self.dim) / 2.0
        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)  # Ensure numerical stability
        return torch.acos(cos_theta)
    
    def deskew(self):
        """
        Extracts the rotation vector from a skew-symmetric cross-product matrix or,
        if the matrix is a full rotation matrix, returns the closest equivalent representation.
        
        If $R$ is a (flattened) cross-product matrix of the vector $w$, this will
        return the vector $w$.  Otherwise, it will return a representation w of
        $\\frac{(R-R^T)}{2}$ (off diagonals of $R$) such that
        $\\frac{(R-R^T)}{2} = \\mathrm{cross_product}(w)$.

        If $R$ is a valid rotation matrix:
          - In 3D, it returns a vector $w$ such that $\\mathrm{cross_product}(w) \\approx (R - R^T)/2$$.
          - In 2D, it returns a single-element vector representing the angle.
          
        Returns
        -------
        Vertex
            A 3D vector $w$ such that $\\mathrm{cross_product}(w) \\approx (R - R^T)/2$$
        """
        
        if self.dim == 3:
            # Compute the skew-symmetric part: (R - R^T) / 2
            skew_part = 0.5 * (self.data - self.data.T)
            return Vertex([skew_part[2, 1], skew_part[0, 2], skew_part[1, 0]])
        elif self.dim == 2:
            # The 'skew' part of a 2D rotation matrix gives sin(theta)
            return Vertex([torch.asin(self.data[1, 0])])  # Simplified for 2D
        else:
            raise ValueError(f"deskew not supported for dimension {self.dim}")
    
    def rotation_vector(self, degrees=False) -> 'Vertex':
        """
        Convert the rotation matrix to a rotation vector (exponential map).
        
        A rotation vector is a vector which is co-directional to the axis of rotation and whose norm
        gives the angle of rotation.
        Specifically, the rotation vector $w$ of a rotation matrix $R$ is such that:
        $\\mathrm{exp}(w) = R$. This is equivalent to data-angle representation with
        $\\frac{w}{||w||}=\\mathrm{data}$ and ${||w||}=\\mathrm{angle}$
        
        Parameters
        ----------
        degrees: bool, optional
            If this flag is set to `True`, the returned magnitudes are in degrees, otherwise in radians.
            Default is `False`.
        Returns
        -------
        Vertex
            A rotation vector (axis-angle representation).
        """
        if self.dim == 3:
            angle = self.angle()
            if angle == 0:
                return Vertex(torch.zeros(3, device=self.device))  # Handle identity matrix case
            
            axis = self.deskew().data / torch.sin(angle)
            rot_vec = axis * angle
            return Vertex(rot_vec)
        elif self.dim == 2:
            angle = self.angle()
            if degrees:
                angle = torch.rad2deg(angle)
            return Vertex([angle])
        else:
            raise ValueError(f"rotation_vector is not supported for dimension {self.dim}")
        
        # r = R.from_matrix(self.array)
        # # return torch.tensor(r.as_rotvec(degrees=degrees), dtype=torch.float32, device=self.device)
        # return Vertex(r.as_rotvec(degrees=degrees))
    
    def distance(self, other: Optional[Union[torch.Tensor, ArrayLike, 'RotationMatrix', list, tuple]]) -> torch.Tensor:
        """
        Compute the geodesic distance between two rotation matrices.

        Parameters
        ----------
        other : RotationMatrix
            Another rotation matrix.

        Returns
        -------
        torch.Tensor
            The geodesic distance.
        """
        other = self.check(other)
        r_diff = self.inverse() @ other
        return r_diff.angle()
    
    @staticmethod
    def from_rotation_vector(rot_vec) -> 'RotationMatrix':
        """
        Create a rotation matrix from a rotation vector (exponential map).

        Parameters
        ----------
        rot_vec : Union[ArrayLike, torch.Tensor, list, tuple]
            A rotation vector (axis-angle representation).

        Returns
        -------
        RotationMatrix
            The rotation matrix.
        """
        rot_vec = torch.as_tensor(rot_vec, dtype=torch.float32, device=torch.device("cuda" if USE_CUDA else "cpu"))
        
        if rot_vec.shape[0] == 3:
            angle = torch.linalg.norm(rot_vec)
            if angle == 0:
                return RotationMatrix(torch.eye(3, device=rot_vec.device))  # Handle zero angle case
            
            axis = rot_vec / angle
            a = torch.cos(angle)
            b = torch.sin(angle)
            v = axis.reshape(3, 1)
            skew_sym = torch.tensor([[0, -v[2, 0], v[1, 0]],
                                     [v[2, 0], 0, -v[0, 0]],
                                     [-v[1, 0], v[0, 0], 0]], device=rot_vec.device)
            R = a * torch.eye(3, device=rot_vec.device) + b * skew_sym + (1 - a) * (v @ v.T)
            return RotationMatrix(R)
        elif rot_vec.shape[0] == 1:
            # 2D rotation from angle
            angle = rot_vec[0]
            R = torch.tensor([[torch.cos(angle), -torch.sin(angle)],
                              [torch.sin(angle), torch.cos(angle)]], device=rot_vec.device)
            return RotationMatrix(R, dim=2)
        else:
            raise ValueError(f"Invalid rotation vector dimension: {rot_vec.shape[0]}")
    
    @staticmethod
    def from_quaternion(quaternion) -> 'RotationMatrix':
        """
        Create a rotation matrix from quaternion representation.

        Parameters
        ----------
        quaternion : array_like
            Quaternion as [x, y, z, w].

        Returns
        -------
        RotationMatrix
        
        References
        ----------
        [1] https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
        [2] https://en.wikipedia.org/wiki/Quaternion
        """
        quaternion = torch.as_tensor(quaternion, dtype=torch.float32,
                                     device=torch.device("cuda" if USE_CUDA else "cpu"))
        x, y, z, w = quaternion
        Nq = w * w + x * x + y * y + z * z
        if Nq < torch.finfo(torch.float32).eps:
            return RotationMatrix(torch.eye(3, device=quaternion.device))
        
        s = 2.0 / Nq
        X = x * s
        Y = y * s
        Z = z * s
        wX = w * X
        wY = w * Y
        wZ = w * Z
        xX = x * X
        xY = x * Y
        xZ = x * Z
        yY = y * Y
        yZ = y * Z
        zZ = z * Z
        
        R = torch.tensor(
            [[1.0 - (yY + zZ), xY - wZ, xZ + wY],
             [xY + wZ, 1.0 - (xX + zZ), yZ - wX],
             [xZ - wY, yZ + wX, 1.0 - (xX + yY)]], device=quaternion.device
        )
        return RotationMatrix(R)
    
    @staticmethod
    def from_euler(angles: Union[ArrayLike, torch.Tensor, list, tuple], order: str = 'xyz',
                   degrees: bool = False) -> 'RotationMatrix':
        """Create a rotation matrix from Euler angles.

        Parameters
        ----------
        angles : Union[ArrayLike, torch.Tensor, list, tuple]
            Euler angles in radians (or degrees if `degrees=True`).
        order : str, optional
            The order of rotations (default is 'xyz').
            Specifies sequence of axes for rotations. Up to 3 characters
            belonging to the set {'X', 'Y', 'Z'} may be specified, where
            each character indicates the axis about which to perform the
            rotation. If a lowercase character is specified, the rotation
            is about the axis of the intrinsic frame.
        degrees : bool, optional
            If `True`, then the given angles are assumed to be in degrees. Default is `False`.

        Returns
        -------
        RotationMatrix
            The rotation matrix.
        """
        
        angles = torch.as_tensor(angles, dtype=torch.float32, device=torch.device("cuda" if USE_CUDA else "cpu"))
        if degrees:
            angles = torch.deg2rad(angles)
        
        if order in ('z', 'y', 'x') and angles.shape[0] == 1 and angles.numel() == 1:
            if order == 'z':
                R = torch.tensor([[torch.cos(angles[0]), -torch.sin(angles[0])],
                                  [torch.sin(angles[0]), torch.cos(angles[0])]], device=angles.device)
                return RotationMatrix(R, dim=2)
            elif order == 'y':
                R = torch.tensor([[torch.cos(angles[0]), torch.sin(angles[0])],
                                  [-torch.sin(angles[0]), torch.cos(angles[0])]], device=angles.device)
                return RotationMatrix(R, dim=2)
            elif order == 'x':  # for similarity and consistency
                R = torch.tensor([[torch.cos(angles[0]), -torch.sin(angles[0])],
                                  [torch.sin(angles[0]), torch.cos(angles[0])]], device=angles.device)
                return RotationMatrix(R, dim=2)
        else:
            
            def _axis_angle_rot(axis: str, angle: torch.Tensor):
                if axis == 'x':
                    rot = torch.tensor([[1, 0, 0],
                                        [0, torch.cos(angle), -torch.sin(angle)],
                                        [0, torch.sin(angle), torch.cos(angle)]],
                                       device=angles.device)
                elif axis == 'y':
                    rot = torch.tensor([[torch.cos(angle), 0, torch.sin(angle)],
                                        [0, 1, 0],
                                        [-torch.sin(angle), 0, torch.cos(angle)]],
                                       device=angles.device)
                elif axis == 'z':
                    rot = torch.tensor([[torch.cos(angle), -torch.sin(angle), 0],
                                        [torch.sin(angle), torch.cos(angle), 0],
                                        [0, 0, 1]],
                                       device=angles.device)
                else:
                    raise ValueError('invalid rotation axis "{}"'.format(axis))
                return rot
            
            rot = torch.eye(3, device=angles.device)
            for i, a in enumerate(order.lower()):
                rot = rot @ _axis_angle_rot(a, angles[i])
            return RotationMatrix(rot)


class Vertex(VectorBase):
    """A  vertex representation in 2D or 3D space
    
    In geometry, a vertex is like a point.
    Specifically, a vertex is where two or more curves, lines or edges intersect or meet.
    """
    
    def transform(self, rotation_matrix: RotationMatrix, translation: 'Vertex'):
        """
        Transform the vertex using a rotation matrix and translation vector.

        Parameters
        ----------
        rotation_matrix : RotationMatrix
            The rotation matrix.
        translation : Vertex
            The translation vector.

        Returns
        -------
        Vertex
            The transformed point.
        """
        return Vertex(rotation_matrix.data @ self.data + translation.data, dim=self.dim)
    
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
    vectors that can be made equal by scaling. Therefore, a direction is represented here as a _unit vector_ similar
    to an axis. Two vectors that share the same direction are said to be codirectional or equidirectional.
    """
    
    def __init__(self, coordinates=None, dim: int = 3):
        """
        Initialize a [`Direction`][pymagbot.math.geometry_base.Direction].

        Parameters
        ----------
        coordinates : array_like, optional
            The coordinates of the direction vector.
        dim : int, optional
            The dimension of the direction vector (2 or 3).
        """
        super().__init__(data=coordinates, dim=dim)
        self.data = self.data / torch.norm(self.data)
    
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
            raise ValueError("Plotting is only supported for 2D and 3D direction vectors.")
        
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


class RigidTransform(MatrixBase):
    """A rigid transformation (SE(2) or SE(3)) between two frames.

    **Rigid transformations** (or Euclidean transformation) include rotations, translations or any sequence of these.
    Any proper rigid transformation can be decomposed into a rotation followed by a translation.
    The set of proper rigid transformations is called special Euclidean group, denoted $SE(n)$ (e.g. n=2 or 3).
    In `pymagbot`, any primitive is defined in a 3D space:  $\\mathbb{R}^3$.
    
    !!! seealso
        [`RotationMatrix`][pymagbot.math.geometry_base.RotationMatrix] class for details about the rotation.
    """
    
    def __init__(self, rotation=None, translation=None, dim: int = 3):
        """
        Initialize the [`RigidTransform`][pymagbot.math.geometry_base.RigidTransform].

        Parameters
        ----------
        rotation : array_like, optional
            The rotation matrix.
        translation : array_like, optional
            The translation vector.
        dim : int, optional
            The dimension of the rigid transformation (2 or 3).
        """
        self.device = torch.device("cuda" if USE_CUDA else "cpu")
        self.R = RotationMatrix(rotation, dim=dim)
        self.d = Vertex(translation, dim=dim)
        super().__init__(self.to_matrix())
        self.dim = dim
    
    def _validate(self):
        if self.data.shape != (self.dim, self.dim):
            raise ValueError(f"RigidTransform must be an homogeneous matrix with {self.dim - 1} dimensions"
                             f"the shape is {self.data.shape}")
    
    def to_matrix(self) -> torch.Tensor:
        """
        Convert the rigid transformation to a homogeneous transformation matrix.

        Returns
        -------
        torch.Tensor
            The homogeneous transformation matrix (shape (dim+1, dim+1)).
        """
        # if self._data is not None:
        T = torch.eye(self.dim + 1, device=self.device, dtype=torch.float32)
        T[:self.dim, :self.dim] = self.R.data
        T[:self.dim, self.dim] = self.d.data
        return T
    
    def inverse(self) -> 'RigidTransform':
        """
        Calculate the inverse of the rigid transformation.

        Returns
        -------
        RigidTransform
            The inverse rigid transformation.
        """
        r_inv = self.R.inverse()
        return RigidTransform(rotation=r_inv, translation=Vertex(-r_inv.data @ self.d.data), dim=self.dim)
