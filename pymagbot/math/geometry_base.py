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
    $R^T = R^{−1}$ and $\\mathrm{det} R = 1$.
    
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
        Validate that the matrix is a valid rotation matrix (orthogonal).

        Raises
        ------
        ValueError
            If the matrix is not orthogonal.
        """
        if not torch.allclose(self.data.T @ self.data, torch.eye(self.dim, device=self.device), atol=1e-6):
            raise ValueError(f"Rotation matrix must be orthogonal, got {self.data}.")
    
    def inverse(self) -> 'RotationMatrix':
        """
        Calculate the inverse of the rotation matrix.

        Returns
        -------
        RotationMatrix
            The inverse rotation matrix.
        """
        return RotationMatrix(self.data.T, dim=self.dim)


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
        #if self._data is not None:
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
