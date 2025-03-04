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

# import numpy as np
# from scipy.spatial.transform import Rotation as R
import itertools
import weakref
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import torch

from pymagbot._config import USE_CUDA

__all__ = ['RotationMatrix', 'Point', 'Direction', 'RigidTransform', 'Frame']

# Dictionary to track known frames (ensuring uniqueness)
known_frames: weakref.WeakValueDictionary[str, 'Frame'] = weakref.WeakValueDictionary()  # Prevent memory leaks

WORLD_FRAME_NAMES: tuple[str, ...] = ('world', 'Frame_0', 'F0', 'workspace')
"""Known reserved name and aliases for world reference frame"""


class MatrixBase(ABC):
    """Abstract base class for matrix-like objects.

    This class provides a foundation for representing matrices using PyTorch tensors.
    It handles device management (CPU or GPU) and provides basic validation.
    
    Attributes
    ----------
    data : torch.Tensor
        The data stored in the matrix. If None, a zero matrix of shape (dim, dim) or (dim,) is created,
        depending on the subclass. Can be a PyTorch tensor, another [`MatrixBase`][pymagbot.frame.MatrixBase] instance, or a list/tuple of numerical values.
    dim : int, optional
        The dimension of the matrix (e.g. 2 for 2D, 3 for 3D).
    """
    
    def __init__(self, data: Optional[Union[torch.Tensor, 'MatrixBase', list, tuple]] = None, dim: int = 3):
        """
        Initialize the MatrixBase with data or shape.

        Parameters
        ----------
        data : array_like, optional
            The data to initialize the matrix.
        dim : tuple, optional
            The shape of the matrix if data is not provided.
        """
        self.device = torch.device("cuda" if USE_CUDA else "cpu")
        self.dim: int = dim
        self._data = self._initialize_data(data)
        self._validate()
    
    @property
    def data(self) -> torch.Tensor:
        """
        Get the underlying PyTorch tensor data.

        Returns
        -------
        torch.Tensor
            The PyTorch tensor representing the matrix data.
        """
        return self._data
    
    @data.setter
    def data(self, value: Union[torch.Tensor, 'MatrixBase', list, tuple]):
        """
        Set the underlying PyTorch tensor data.

        Parameters
        ----------
        value : Union[torch.Tensor, 'MatrixBase', list, tuple]
            The new data for the matrix.  It will be converted to a PyTorch tensor and moved to the appropriate device.
        """
        self._data = self._initialize_data(value)
        self._validate()
    
    def _initialize_data(self, data) -> torch.Tensor:
        """
        Initialize the PyTorch tensor data from various input types.

        Parameters
        ----------
        data : Union[torch.Tensor, 'MatrixBase', list, tuple], optional
            The input data.

        Returns
        -------
        torch.Tensor
            The initialized PyTorch tensor.
        """
        if data is None:
            shape = (self.dim, self.dim) if isinstance(self, RotationMatrix) else (self.dim,)
            return torch.zeros(shape, device=self.device, dtype=torch.float32)
        
        if isinstance(data, MatrixBase):
            data = data.data
        if torch.is_tensor(data):
            return data.clone().to(self.device).float()  # Ensure it's a float tensor and on the correct device
        return torch.tensor(data, dtype=torch.float32, device=self.device)  # Convert list/tuple to tensor
    
    @abstractmethod
    def _validate(self):
        """
        Validate the shape and properties of the data.
        
        This method must be implemented by subclasses to ensure data consistency.

        Raises
        ------
        ValueError
            If the data does not meet the required criteria (e.g., shape mismatch, non-orthogonal matrix).
        """
        pass
    
    def array(self):
        return self.data.cpu().numpy()
    
    def __str__(self) -> str:
        """
        String representation of the matrix.

        Returns
        -------
        str
            A string representation of the matrix data (CPU numpy array).
        """
        return str()
    
    def __repr__(self) -> str:
        """
        String representation of the object.

        Returns
        -------
        str
            A string representation of the object including class name and data.
        """
        return f"{self.__class__.__name__}({self.array()})"
    
    def to_list(self) -> List[Any]:
        """
        Convert the matrix data to a list.

        Returns
        -------
        list
            A list representation of the matrix data.
        """
        return self.array().tolist()
    
    def to_dict(self) -> Dict[str, List[Any]]:
        """
        Convert the matrix data to a dictionary.

        Returns
        -------
        dict
            A dictionary representation of the matrix data.
        """
        return {"data": self.to_list()}
    
    @classmethod
    def from_dict(cls, data_dict: Dict[str, List[Any]]) -> 'MatrixBase':
        """
        Create an instance of the class from a dictionary.

        Parameters
        ----------
        data_dict : dict
            A dictionary containing the matrix data. Must have a "data" key.

        Returns
        -------
        MatrixBase
            A new instance of the class initialized with the data from the dictionary.
        """
        return cls(data_dict["data"])
    
    def clone(self) -> 'MatrixBase':
        """
        Create a deep copy of the object.

        Returns
        -------
        MatrixBase
            A deep copy of the object.
        """
        return self.__class__(self.data.clone())


class VectorBase(MatrixBase):
    """Base class for vector-like objects, such as  [`Point`][pymagbot.frame.Point] and [`Direction`][pymagbot.frame.Direction]."""
    
    def _validate(self):
        if self.data.shape != (self.dim,):
            raise ValueError(f"Vector must have shape ({self.dim},), got {self.data.shape}.")


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


class Frame():
    """
    An frame defined by a rigid transformation (SE(2) or SE(3)).
    
    This class represents a inertia frame in 2D or 3D space.  It manages the frame's
    transformation relative to its parent frame, and maintains a registry of known frames.
    """
    _id_counter = itertools.count(1)  # Unique ID generator for frames
    
    world = None
    
    def __new__(cls, name: Optional[str] = None, parent: Optional['Frame'] = None,
                transform: Optional[RigidTransform] = None, dim: int = 3) -> 'Frame':
        """
        Create a new Frame instance or retrieve an existing one with the same name.

        This method ensures that only one [`Frame`][pymagbot.frame.Frame] instance exists for each unique name.

        Parameters
        ----------
        name : str, optional
            The name of the frame. If None, a unique name is generated.
        parent : Frame, optional
            The parent frame. If None, the world frame is used as the parent.
        transform : RigidTransform, optional
            The rigid transformation. If None, an identity transformation is used.
        dim : int, optional
            The dimension of the frame (2 or 3). Defaults to 3.

        Returns
        -------
        Frame
            The [`Frame`][pymagbot.frame.Frame] instance.  If a frame with the given name already exists, that instance is returned.
        """
        # name = cls._generate_unique_name(name)
        
        name = name.lower() if name else f"Frame_{next(cls._id_counter)}"
        if name in known_frames:
            return known_frames[name]
        instance = super().__new__(cls)
        instance._name: str = name  # Store the name early
        known_frames[name] = instance
        return instance
    
    def __init__(self, name: Optional[str] = None, parent: Optional['Frame'] = None,
                 transform: Optional[RigidTransform] = None, dim: int = 3):
        """
        Initialize the [`Frame`][pymagbot.frame.Frame].

        Parameters
        ----------
        name : str, optional
            The name of the frame.
        parent : Frame, optional
            The parent frame.
        transform : RigidTransform, optional
            The rigid transformation.
        dim : int, optional
            The dimension of the frame (e.g. 2 or 3).
        """
        
        if hasattr(self, 'initialized'):
            return  # Prevent double initialization
        
        self.name: str = self._name
        self.dim: int = dim
        self.children: set['Frame'] = set()
        self.transform: RigidTransform = transform if transform else RigidTransform(dim=dim)
        
        if self.is_world():
            if Frame.world is None:
                Frame.world = self  # Set the world frame
            elif Frame.world is not self:
                raise ValueError("Only one world frame instance is allowed.")
        
        self.parent: 'Frame' = Frame.world if parent is None and not self.is_world() else parent
        if self.parent:
            self.parent.children.add(self)
        
        self.initialized: bool = True
    
    def __iter__(self):
        yield from self.children
    
    def is_world(self) -> bool:
        """
        Check if the frame is the world frame.

        Returns
        -------
        bool
            True if the frame's name is in the list of world frame names, False otherwise.
        """
        return self.name in WORLD_FRAME_NAMES
    
    is_reference = is_world
    
    def is_child(self, frame) -> bool:
        """
        Check if the given frame is a direct child of this frame.

        Parameters
        ----------
        frame : Frame
            The frame to check.

        Returns
        -------
        bool
            True if the frame is a child, False otherwise.
        """
        return frame in self.children
    
    def is_ancestor(self, frame: 'Frame') -> bool:
        """
        Check if the given frame is an ancestor of this frame.

        Parameters
        ----------
        frame : Frame
            The frame to check.

        Returns
        -------
        bool
            True if the frame is an ancestor, False otherwise.
        """
        parent: Optional['Frame'] = self.parent
        while parent:
            if parent == frame:
                return True
            parent = parent.parent
        return False
    
    def move_to(self, translation, local: bool = True):
        """
        Move the frame to a new location.

        Parameters
        ----------
        translation : array_like
            The translation vector.
        local : bool, optional
            Whether the translation is local to the parent frame or with respect to the world frame.
        """
        if not isinstance(translation, Point):
            translation = Point(translation, dim=self.dim)
        if local and self.parent:
            self.transform.d = Point(self.parent.transform.R.data @ translation.data + self.parent.transform.d.data,
                                     dim=self.dim)
        else:
            self.transform.d = translation  # Set the translation directly in world coordinates
        self._propagate_changes()
    
    def rotate(self, rotation_matrix, local: bool = True):
        """
        Rotate the frame.

        Parameters
        ----------
        rotation_matrix : array_like
            The rotation matrix.
        local : bool, optional
            Whether the rotation is local to the parent frame or with respect to the world frame.
        """
        if not isinstance(rotation_matrix, RotationMatrix):
            rotation_matrix = RotationMatrix(rotation_matrix, dim=self.dim)
        if local and self.parent:
            self.transform.R = RotationMatrix(rotation_matrix.data @ self.transform.R.data, dim=self.dim)
        else:
            self.transform.R = rotation_matrix
        self._propagate_changes()
    
    def _propagate_changes(self):
        """Propagate the changes to the transform to all children."""
        for child in self.children:
            # Update child's translation and rotation based on parent's transformation
            child.transform.d = Point(self.transform.R.data @ child.transform.d.data + self.transform.d.data,
                                      dim=self.dim)
            child.transform.R = RotationMatrix(self.transform.R.data @ child.transform.R.data, dim=self.dim)
            child._propagate_changes()
    
    def remove(self):
        """
        Remove the frame and its children from the known frames list.
        """
        if self.name in WORLD_FRAME_NAMES:
            raise ValueError(f"Cannot remove '{self.name}'!")
        # Remove the frame and its children from the known frames list
        del known_frames[self.name]
        for child in self.children:
            child.remove()
        
        # Remove the frame from its parent's children list
        if self.parent:
            self.parent.children.discard(self)
    
    def __enter__(self):
        """
        Enter the context management.

        Returns
        -------
        Frame
            The frame instance.
        """
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        """
        Exit the context management and remove the frame.
        """
        self.remove()
    
    def inverse(self) -> 'Frame':
        """
        Create a new frame that represents the inverse transformation of this frame.

        Returns
        -------
        Frame
            A new [`Frame`][pymagbot.frame.Frame] instance with the inverse transformation.
        """
        return Frame(name=self.name + "_inv", parent=self.parent, transform=self.transform.inverse(), dim=self.dim)
    
    def to_matrix(self):
        return self.transform.to_matrix()
    
    def __repr__(self):
        return f"name={self.name}, Frame(transform={self.transform.to_matrix().cpu().numpy()}, parent={self.parent})"
    
    @staticmethod
    def known() -> weakref.WeakValueDictionary:
        """
        Get the dictionary of known frames.

        Returns
        -------
        weakref.WeakValueDictionary
            The dictionary containing all known frames.
        """
        return known_frames
    
    def plot(self, scene=None, scale=1.0, label=True, name=None):
        """Plot the frame using Plotly.

        Parameters
        ----------
        scene : plotly.graph_objects.Scene, optional
            The Plotly scene to add the frame to. If None, a new figure and scene will be created.
        scale : float, optional
            The scale of the coordinate frame axes. Defaults to 1.0.
        label : bool, optional
            Whether to display the frame name as a label. Defaults to True.

        Returns
        -------
        plotly.graph_objects.Figure or plotly.graph_objects.Scene
            The Plotly figure object, or the modified scene if a scene was provided as input.

        Raises
        ------
        ImportError
            If Plotly is not installed.
        ValueError
            If the frame is not 3D.
        """
        try:
            import plotly.graph_objects as go
            import numpy as np  # Import numpy within the try block
        except ImportError:
            raise ImportError("Plotly is required for plotting.")
        
        if self.dim != 3:
            raise ValueError("Plotting is only supported for 3D frames.")
        
        T = self.to_matrix().cpu().numpy()
        origin = T[:3, 3]
        R = T[:3, :3]
        
        x_axis = R[:, 0] * scale
        y_axis = R[:, 1] * scale
        z_axis = R[:, 2] * scale
        
        if scene is None:
            fig = go.Figure()
            scene = go.Scene(
                xaxis=dict(title='X'),
                yaxis=dict(title='Y'),
                zaxis=dict(title='Z'),
                aspectmode='data'  # Ensures proper scaling
            )
            fig.update_layout(scene=scene)  # Assign the scene to the layout
        
        # Plot the axes
        origin_point = Point(data=torch.tensor(origin), dim=3)
        x_direction = Direction(data=torch.tensor(x_axis), dim=3)
        y_direction = Direction(data=torch.tensor(y_axis), dim=3)
        z_direction = Direction(data=torch.tensor(z_axis), dim=3)
        
        origin_point.plot(scene=scene, color='black', label=name if name else self.name + "_origin", size=4)
        x_direction.plot(scene=scene, origin=origin_point, color='red', scale=1,
                         label=name if name else self.name + "_X")
        y_direction.plot(scene=scene, origin=origin_point, color='green', scale=1,
                         label=name if name else self.name + "_Y")
        z_direction.plot(scene=scene, origin=origin_point, color='blue', scale=1,
                         label=name if name else self.name + "_Z")
        
        if scene is None:
            return fig
        else:
            return scene
    
    def plot_tree(self, scene=None, scale=1.0, name=None):
        """Plot the frame and all its children as a tree structure.

        Parameters
        ----------
        scene : plotly.graph_objects.Scene, optional
            The Plotly scene to add the frame to. If None, a new figure and scene will be created.
        scale : float, optional
            The scale of the coordinate frame axes. Defaults to 1.0.

        Returns
        -------
        plotly.graph_objects.Figure
            The Plotly figure object.
        """
        
        if scene is None:
            fig = go.Figure()
            scene = go.Scene(
                xaxis=dict(title='X'),
                yaxis=dict(title='Y'),
                zaxis=dict(title='Z'),
                aspectmode='data'  # Ensures proper scaling
            )
            fig.update_layout(scene=scene)
        
        scene = self.plot(scene=scene, scale=scale, name=name if name else self.name)
        for i, child in enumerate(self.children):
            child.plot_tree(scene=scene, scale=scale, name=name if name else self.name + f"_{i + 1}")
        if scene is None:
            return fig
        else:
            return fig


# Initialize the world frame
world = Frame(name=WORLD_FRAME_NAMES[0])
