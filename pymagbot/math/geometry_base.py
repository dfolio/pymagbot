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
from typing import Optional, Union

import torch

from pymagbot._config import USE_CUDA
from pymagbot.math.matrix_base import MatrixBase, VectorBase

__all__ = ['RotationMatrix', 'Point', 'Direction', 'RigidTransform']


class RotationMatrix(MatrixBase):
    """A  rotation matrix representing an element of  SO(2) or SO(3)."""
    
    def __init__(self, matrix: Optional[Union[torch.Tensor, 'RotationMatrix', list, tuple]] = None, dim: int = 3):
        """
        Initialize the RotationMatrix.

        Parameters
        ----------
        matrix : array_like, optional
            The rotation matrix data.
        dim : int, optional
            The dimension of the rotation matrix (2 or 3).
        """
        super().__init__(matrix if matrix is not None else torch.eye(dim), dim=dim)
    
    def _validate(self) -> 'RotationMatrix':
        """
        Validate that the matrix is a valid rotation matrix (orthogonal).

        Raises
        ------
        ValueError
            If the matrix is not orthogonal.
        """
        if not torch.allclose(self.data.T @ self.data, torch.eye(self.dim, device=self.device), atol=1e-6):
            raise ValueError(f"Rotation matrix must be orthogonal, got {self.data}.")
    
    def inverse(self):
        """
        Calculate the inverse of the rotation matrix.

        Returns
        -------
        RotationMatrix
            The inverse rotation matrix.
        """
        return RotationMatrix(self.data.T, dim=self.dim)


class Point(VectorBase):
    """A  point representation in in 2D or 3D space"""
    
    def transform(self, rotation_matrix: RotationMatrix, translation: 'Point'):
        """
        Transform the point using a rotation matrix and translation vector.

        Parameters
        ----------
        rotation_matrix : RotationMatrix
            The rotation matrix.
        translation : Point
            The translation vector.

        Returns
        -------
        Point
            The transformed point.
        """
        return Point(rotation_matrix.data @ self.data + translation.data, dim=self.dim)
    
    def plot(self, scene=None, color='k', marker='o', size=5, label=None):
        """Plot the point using Plotly.

        Parameters
        ----------
        scene : plotly.graph_objects.Scene, optional
            The Plotly scene to add the point to. If None, a new figure and scene will be created.
        color : str, optional
            The color of the point. Defaults to 'k' (black).
        marker : str, optional
            The marker style. Defaults to 'o' (circle).
        size : int, optional
            The size of the marker. Defaults to 5.
        label : str, optional
            The label to display for the point. If None, no label will be displayed.

        Returns
        -------
        plotly.graph_objects.Figure
            The Plotly figure object.

        Raises
        ------
        ImportError
            If Plotly is not installed.
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
        
        if scene is None:
            fig = go.Figure(data=[trace])
            return fig
        else:
            scene.add_trace(trace)
            return scene


class Direction(VectorBase):
    """
    A direction vector in 2D or 3D, which must be a unit vector.
    """
    
    def __init__(self, coordinates=None, dim: int = 3):
        """
        Initialize the Direction.

        Parameters
        ----------
        coordinates : array_like, optional
            The coordinates of the direction vector.
        dim : int, optional
            The dimension of the direction vector (2 or 3).
        """
        super().__init__(coordinates, dim=dim)
        self.data = self.data / torch.norm(self.data)
    
    def plot(self, scene=None, origin=None, color='r', scale=1.0, label=None):
        """Plot the direction vector using Plotly.

        Parameters
        ----------
        scene : plotly.graph_objects.Scene, optional
            The Plotly scene to add the direction vector to. If None, a new figure and scene will be created.
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
            The Plotly figure object.

        Raises
        ------
        ImportError
            If Plotly is not installed.
        ValueError
            If the direction vector is not 2D or 3D.
        """
        try:
            import plotly.graph_objects as go
            import numpy as np
        except ImportError:
            raise ImportError("Plotly is required for plotting. Install with `pip install plotly`.")
        
        if self.dim not in [2, 3]:
            raise ValueError("Plotting is only supported for 2D and 3D direction vectors.")
        
        if origin is None:
            origin = Point(torch.zeros(self.dim), dim=self.dim)
        
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
        
        if scene is None:
            fig = go.Figure(data=[trace])
            return fig
        else:
            scene.add_trace(trace)
            return scene


class RigidTransform:
    """A rigid transformation (SE(2) or SE(3)) between two frames.

    This class represents a rigid transformation, which includes both a rotation and a translation.
    """
    
    def __init__(self, rotation=None, translation=None, dim: int = 3):
        """
        Initialize the RigidTransform.

        Parameters
        ----------
        rotation : array_like, optional
            The rotation matrix.
        translation : array_like, optional
            The translation vector.
        dim : int, optional
            The dimension of the transformation (2 or 3).
        """
        self.device = torch.device("cuda" if USE_CUDA else "cpu")
        self.dim: int = dim
        self.R = RotationMatrix(rotation, dim=dim)
        self.d = Point(translation, dim=dim)
    
    def to_matrix(self) -> torch.Tensor:
        """
        Convert the rigid transformation to a homogeneous transformation matrix.

        Returns
        -------
        torch.Tensor
            The homogeneous transformation matrix (shape (dim+1, dim+1)).
        """
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
        R_inv = self.R.inverse()
        d_inv = Point(-R_inv.data @ self.d.data)
        # return RigidTransform(rotation=R_inv.data.cpu().numpy(), translation=d_inv.data.cpu().numpy())
        
        return RigidTransform(rotation=R_inv, translation=d_inv)
