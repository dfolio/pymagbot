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

# import numpy as np
# from scipy.spatial.transform import Rotation as R
import itertools
from abc import ABC, abstractmethod
import torch
import weakref

from pymagbot.config import USE_CUDA

__all__ = ['RotationMatrix', 'Point', 'Direction', 'RigidTransform', 'Frame']

# Dictionary to track known frames (ensuring uniqueness)
known_frames = weakref.WeakValueDictionary()  # Prevent memory leaks

WORLD_FRAME_NAMES = ('world', 'Frame_0', 'F0', 'workspace')
"""Known reserved name and aliases for world reference frame"""


class MatrixBase(ABC):
    """Abstract base class for matrix-like objects.

    Attributes
    ----------
    data : torch.Tensor
        The data stored in the matrix.
    dim : int
        The dimension of the matrix (e.g. 2 for 2D, 3 for 3D).
    device : torch.device
        The device (CPU or GPU) where the data is stored.
    """
    
    def __init__(self, data=None, dim: int = 3):
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
        self.dim = dim
        self._data = self._initialize_data(data)
        self._validate()
    
    @property
    def data(self):
        return self._data
    
    @data.setter
    def data(self, value):
        self._data = self._initialize_data(value)
    
    def _initialize_data(self, data):
        if data is None:
            return torch.zeros((self.dim, self.dim) if isinstance(self, RotationMatrix) else (self.dim,),
                               device=self.device)
        
        if isinstance(data, MatrixBase):
            data = data.data
        
        if torch.is_tensor(data):
            return data.clone().to(self.device)
        return torch.tensor(data, dtype=torch.float32, device=self.device)
    
    @abstractmethod
    def _validate(self):
        """
        Validate the shape and properties of the data.
        """
        pass
    
    def __str__(self):
        return str(self.data.cpu().numpy())
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self.data.cpu().numpy()})"
    
    def to_list(self):
        return self.data.cpu().numpy().tolist()
    
    def to_dict(self):
        return {"data": self.to_list()}
    
    @classmethod
    def from_dict(cls, data_dict):
        return cls(data_dict["data"])
    
    def __add__(self, other):
        return self.__class__(self.data + other.data)
    
    def __sub__(self, other):
        return self.__class__(self.data - other.data)
    
    def __matmul__(self, other):
        return self.__class__(self.data @ other.data)
    
    def __mul__(self, scalar):
        return self.__class__(self.data * scalar)
    
    def __truediv__(self, scalar):
        return self.__class__(self.data / scalar)
    
    def clone(self):
        """
        Create a deep copy of the object.

        Returns
        -------
        MatrixBase
            A deep copy of the object.
        """
        return self.__class__(self.data.clone())


class RotationMatrix(MatrixBase):
    """A  rotation matrix representing an element of  SO(2) or SO(3)."""
    
    def __init__(self, matrix=None, dim: int = 3):
        """
        Initialize the RotationMatrix.

        Parameters
        ----------
        matrix : array_like, optional
            The rotation matrix data.
        dim : int, optional
            The dimension of the rotation matrix (2 or 3).
        """
        if matrix is None:
            matrix = torch.eye(dim)
        super().__init__(matrix, dim=dim)
    
    def _validate(self):
        # or
        #                 not torch.isclose(torch.det(self.data), torch.tensor(1.0, device=self.device), atol=1e-6)
        if (self.data.shape != (self.dim, self.dim) or
                not torch.allclose(self.data.T @ self.data, torch.eye(self.dim, device=self.device), atol=1e-6) ):
            raise ValueError(f"Rotation matrix must be orthogonal and of shape ({self.dim},{self.dim}) get {self.data.shape}.")
    
    def inverse(self):
        return RotationMatrix(self.data.T.cpu().numpy(), dim=self.dim)


class Point(MatrixBase):
    """A  point representation."""
    
    def __init__(self, coordinates=None, dim: int = 3):
        """
        Initialize the Point.

        Parameters
        ----------
        coordinates : array_like, optional
            The coordinates of the point.
        dim : int, optional
            The dimension of the point (2 or 3).
        """
        super().__init__(coordinates, dim=dim)
    
    def _validate(self):
        if self.data.shape != (self.dim,):
            raise ValueError(f"Point must have shape ({self.dim},),).")
    
    def transform(self, rotation_matrix, translation):
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


class Direction(Point):
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


class RigidTransform:
    """A rigid transformation (SE(2) or SE(3)) between two frames."""
    
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
        self.dim = dim
        self.R = RotationMatrix(rotation, dim=dim)
        self.d = Point(translation, dim=dim)
    
    def to_matrix(self):
        T = torch.eye(self.dim + 1, device=self.device)
        T[:self.dim, :self.dim] = self.R.data
        T[:self.dim, self.dim] = self.d.data
        return T
    
    def inverse(self):
        R_inv = self.R.inverse()
        d_inv = Point(-R_inv.data @ self.d.data)
        return RigidTransform(rotation=R_inv.data.cpu().numpy(), translation=d_inv.data.cpu().numpy())


class AbstractFrame(ABC):
    """
    Abstract base class for frames.
    """
    
    @property
    @abstractmethod
    def name(self):
        """
        Abstract property for the frame name.
        """
        pass
    
    @property
    @abstractmethod
    def parent(self):
        """
        Abstract property for the parent frame.
        """
        pass
    
    @property
    @abstractmethod
    def transform(self):
        """
        Abstract property for the frame transformation.
        """
        pass
    
    @abstractmethod
    def move_to(self, translation, local=True):
        """
        Abstract method to move the frame to a new translation.
        """
        pass
    
    @abstractmethod
    def rotate(self, rotation_matrix, local=True):
        """
        Abstract method to rotate the frame.
        """
        pass
    
    @abstractmethod
    def remove(self):
        """
        Abstract method to remove the frame.
        """
        pass
    
    @abstractmethod
    def inverse(self):
        """
        Abstract method to get the inverse of the frame.
        """
        pass
    
    @abstractmethod
    def to_matrix(self):
        """
        Abstract method to convert the frame to a transformation matrix.
        """
        pass


class Frame():
    """
    A reference frame defined by a rigid transformation (SE(2) or SE(3)).
    """
    _id_counter = itertools.count(1)  # Unique ID generator for frames
    
    world_frame = None
    
    def __new__(cls, name: str = "world", parent=None, transform: RigidTransform = None, dim: int = 3):
        #name = cls._generate_unique_name(name)
        if name is None:
            name = f"Frame_{next(self._id_counter)}"
        elif name in WORLD_FRAME_NAMES:
            name = WORLD_FRAME_NAMES[0]
        else:
          # Ensure the name is unique and case-insensitive
          existing_names = {frame.lower() for frame in known_frames.keys()}
          base_name = name.lower()
          counter = 1
          unique_name = base_name
          
          while unique_name in existing_names:
              unique_name = f"{base_name}_{counter}"
              counter += 1
          
          name = unique_name        
        
        if name in known_frames:
            return known_frames[name]
        instance = super().__new__(cls)
        known_frames[name] = instance
        return instance
    
    def __init__(self, name: str = "world", parent=None, transform: RigidTransform = None, dim: int = 3):
        """
        Initialize the Frame.

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
            return
        
        self.name = name
        if self.is_world_frame() and Frame.world_frame is None:
            Frame.world_frame = self
        elif self.is_world_frame():
            raise ValueError("Only one world_frame instance is allowed.")
        
        if parent:
            if isinstance(parent, str):
                parent = self.known_frames.get(parent.lower())
            if not isinstance(parent, Frame):
                raise ValueError("Parent must be an instance of Frame or None.")
        self.parent = parent 
        self.children = set()
        if self.parent:
            self.parent.children.add(self)
        
        self.dim = dim
        self.transform = RigidTransform(dim=dim) if transform is None else transform
        self.initialized = True
    
    def __iter__(self):
        return iter(self.children)
    
    def is_world_frame(self) -> bool:
        return self.name in WORLD_FRAME_NAMES
    
    is_reference = is_world_frame
    
    def is_child(self, frame):
        return frame in self.children
    
    def is_ancestor(self, frame):
        parent = self.parent
        while parent:
            if parent == frame:
                return True
            parent = parent.parent
        return False
    
    def move_to(self, translation, local=True):
        """
        Move the frame to a new translation.

        Parameters
        ----------
        translation : array_like
            The translation vector.
        local : bool, optional
            Whether the translation is local to the parent frame or with respect to the world frame.
        """
        translation = Point(translation)
        if local and self.parent:
            self.transform.d = Point(self.parent.transform.R.data @ translation.data + self.parent.transform.d.data,
                                     dim=self.dim)
        else:
            self.transform.d = translation
        self._propagate_changes()
    
    def rotate(self, rotation_matrix, local=True):
        """
        Rotate the frame.

        Parameters
        ----------
        rotation_matrix : array_like
            The rotation matrix.
        local : bool, optional
            Whether the rotation is local to the parent frame or with respect to the world frame.
        """
        rotation_matrix = RotationMatrix(rotation_matrix, dim=self.dim)
        if local and self.parent:
            self.transform.R = RotationMatrix(rotation_matrix.data @ self.transform.R.data,
                                              dim=self.dim, use_gpu=(self.device.type == "cuda"))
        else:
            self.transform.R = rotation_matrix
        self._propagate_changes()
    
    def _propagate_changes(self):
        for child in self.children:
            child.transform.d = Point(self.transform.R.data @ child.transform.d.data + self.transform.d.data,
                                      dim=self.dim, use_gpu=(self.device.type == "cuda"))
            child.transform.R = RotationMatrix(self.transform.R.data @ child.transform.R.data,
                                               dim=self.dim, use_gpu=(self.device.type == "cuda"))
            child._propagate_changes()
    
    def remove(self):
        """
        Remove the frame and its children from the known frames list.
        """
        if self.name.lower() in WORLD_FRAME_NAMES:
            raise ValueError(f"The reference frame '{self.name}' cannot be removed.")
        
        # Remove the frame and its children from the known frames list
        known_frames[:] = [frame for frame in known_frames if frame.name != self.name]
        for child in self.children:
            child.remove()
        
        # Remove the frame from its parent's children list
        if self.parent:
            self.parent.children.remove(self)
    
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
    
    def inverse(self):
        return Frame(parent=self.parent, transform=self.transform.inverse(), dim=self.dim,
                     use_gpu=(self.device.type == "cuda"))
    
    def to_matrix(self):
        return self.transform.to_matrix()
    
    def __repr__(self):
        return f"name={self.name}, Frame(transform={self.transform.to_matrix().cpu().numpy()}, parent={self.parent})"


# Initialize the world frame
world_frame = Frame(name="world")
