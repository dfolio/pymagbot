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
spatial - Submodule for base spatial transformations.

This module provides spatial transformations for **rigid body motion** in $SE(2)$ and $SE(3)$.
It includes classes for rotation matrices and rigid transformations.

Tip
---
In `pymagbot`  **both 2D and 3D transformations** are supported.

Examples
--------
>>> from pymagbot.math.spatial import RotationMatrix, RigidTransform
>>> _rotation = RotationMatrix.from_euler([0.1, 0.2, 0.3], order='xyz')
>>> T = RigidTransform(_rotation,[1, 2, 3])
>>> print(T.as_matrix())
"""

from abc import abstractmethod
from typing import Optional, Union

import torch
from numpy.typing import ArrayLike

from pymagbot.utils.config import DEVICE, DTYPE
from pymagbot.math.matrix_base import ensure_tensor, MatrixBase, parse_dict
from pymagbot.math.spatial.primitives import Direction, VectorBase, Vertex
from pymagbot.math.spatial.operations import (euler_to_rotation, quaternion_to_rotation, rotation_to_euler,
                                              rotation_to_quaternion, vector_to_rotation)

__all__ = ['RotationMatrix', 'RigidTransform']


def safe_vector_norm(vector: torch.Tensor) -> torch.Tensor:
    """Calculate the norm of a vector with handling for zero components.

    Returns
    -------
    torch.Tensor
        The norm of the vector.  Returns 0 for zero components to avoid division by zero.
    """
    norm = torch.linalg.norm(vector)
    return norm if norm > torch.finfo(DTYPE).eps else torch.tensor(0.0, dtype=DTYPE, device=vector.device)


def create_unit_vector(vector: torch.Tensor) -> Union[Direction, Vertex]:
    norm = safe_vector_norm(vector)
    if norm <= 0.:
        # print("Warning:  The vector is an zero vector with small Norm.")
        return Vertex(torch.zeros_like(vector))
    else:
        return Direction(vector / norm)


class TransformBase(MatrixBase):
    """Abstract base class for transformation objects.
    
    This class provides a common interface for transformation objects,
    defining abstract methods for obtaining the transformation matrix and
    its inverse.
    """
    
    @abstractmethod
    def as_matrix(self):
        """Return the transformation matrix."""
        pass
    
    # as_tensor = as_matrix
    
    @abstractmethod
    def inverse(self):
        """Return the inverse transformation."""
        pass


class RotationMatrix(TransformBase):
    """A  rotation matrix representing an element of  SO(2) or SO(3).
    
    In linear algebra, a rotation matrix is a transformation matrix that is used to perform a rotation in Euclidean
    space. More specifically, they can be characterized as **orthogonal matrice**, that is, a _square matrix_ $R$ is a
    rotation matrix if and only if
    $R^T = R^{−1}$ and $\\mathrm{det}(R) = \pm1$.
    
    The set of all orthogonal matrices of size $n$ is a representation of a group known as the
    _special orthogonal group_ $SO(n)$ (here $n=$2 or 3),
    Every rotation maps an orthonormal basis of $\\mathbb{R}^n$ to another orthonormal basis.
    
    Attributes
    ----------
    data : torch.Tensor
        The rotation matrix data.
    dim : int
        The dimension of the rotation matrix (2 or 3).
        
    References
    ----------
    - <https://en.wikipedia.org/wiki/Rotation_matrix>
    """
    
    def __init__(self, matrix: Optional[Union[torch.Tensor, ArrayLike, 'RotationMatrix', list, tuple]] = None,
                 dim: int = None):
        """Initialize the [`RotationMatrix`][pymagbot.math.spatial.RotationMatrix].

        Parameters
        ----------
        matrix : array_like, optional
            The rotation matrix data. If `None`, an identity matrix is created
        dim : int, optional
            The dimension of the rotation matrix (2 or 3).
        """
        super().__init__(matrix, dim=dim)
    
    def _initialize_data(self, matrix):
        if matrix is None:
            return torch.eye(self.dim, dtype=DTYPE, device=self.device)
        if isinstance(matrix, MatrixBase):
            return matrix.data.clone()
        if isinstance(matrix, torch.Tensor):
            return matrix
        return torch.as_tensor(matrix, dtype=DTYPE, device=self.device)
    
    def _validate(self):
        """Ensure that the matrix is a valid rotation matrix (orthogonal and determinant=1).

        Raises
        ------
        ValueError
            If the matrix is not orthogonal or the determinant is not close to 1.
        """
        if self.data.shape != (self.dim, self.dim):
            raise ValueError(f"Rotation matrix must be square with dimension {self.dim}, got {self.data.shape}.")
        if not self.is_orthogonal():
            raise ValueError(f"Rotation matrix must be orthogonal, got {self.array}.")
    
    def is_orthogonal(self, atol=1e-6) -> bool:
        """Check if the matrix is orthogonal."""
        rot_mat = self.data
        return (torch.allclose(rot_mat.T @ rot_mat, torch.eye(self.dim, dtype=DTYPE, device=self.device), atol=atol) and
                torch.isclose(torch.abs(torch.det(rot_mat)),
                              torch.tensor(1.0, dtype=DTYPE, device=self.device), atol=atol))
    
    def as_matrix(self) -> torch.Tensor:
        """Return the rotation matrix."""
        return self.data
    
    @property
    def rotation(self) -> torch.Tensor:
        """Return the rotation matrix, already cast to tensor"""
        return self.as_matrix()
    
    @classmethod
    def check(cls, matrix: Union[torch.Tensor, ArrayLike, 'RotationMatrix', list, tuple]) -> 'RotationMatrix':
        """Check if rotation is or can be converted to [`RotationMatrix`][pymagbot.math.spatial.RotationMatrix]

        Parameters
        ----------
        matrix : Union[torch.Tensor, ArrayLike, RotationMatrix, list, tuple]
            The rotation to check or convert.

        Returns
        -------
        RotationMatrix
            The checked [`RotationMatrix`][pymagbot.math.spatial.RotationMatrix].

        Raises
        ------
        TypeError
            If the input cannot be converted to a RotationMatrix.
        """
        if isinstance(matrix, (torch.Tensor, ArrayLike, list, tuple)):
            matrix = cls(matrix)
        if not isinstance(matrix, RotationMatrix):
            raise TypeError('matrix is not a valid rotation matrix.')
        return matrix
    
    def __matmul__(self, other):
        """Matrix multiplication between [`RotationMatrix`][pymagbot.math.spatial.RotationMatrix] and an `other` objects.
        
        !!! note
            - If `other` is a [`RotationMatrix`][pymagbot.math.spatial.RotationMatrix], the result is a new rotation
              matrix.
            - If `other` is a [`VectorBase`][pymagbot.math.spatial.primitives.VectorBase], the result is a `Vertex`.
            - If `other` is a [`RigidTransform`][pymagbot.math.spatial.RigidTransform], the result is a `RigidTransform`.
            - Otherwise, if `other` can be converted to a [`MatrixBase`][pymagbot.math.matrix_base.MatrixBase],
              the result is a matrix.

        Parameters
        ----------
        other : array_like
            The other data object to multiply.

        Returns
        -------
        MatrixBase
            The result of the [`MatrixBase`][pymagbot.math.matrix_base.MatrixBase] multiplication.
        """
        if isinstance(other, RotationMatrix):
            return self.__class__(self.data @ other.data)
        if isinstance(other, VectorBase):
            return Vertex(self.data @ other.data)
        if isinstance(other, RigidTransform):
            return RigidTransform(rotation=self.data @ other.rotation.data, translation=other.translation)
        try:
            other = ensure_tensor(other)
            output: MatrixBase = super().__matmul__(other)
            return output
        except Exception as E:
            raise TypeError(f"Unsupported operation between RotationMatrix and {type(other)}: {E}")
    
    @classmethod
    def identity(cls, dim: int = None) -> 'RotationMatrix':
        """Return the identity rotation matrix.

        Parameters
        ----------
        dim : int, optional
            The dimension of the rotation matrix (2 or 3). Default is 3.

        Returns
        -------
        RotationMatrix
            The identity rotation matrix.
        """
        dim = dim if not None else 3
        return cls(torch.eye(dim, dtype=DTYPE, device=DEVICE), dim=dim)
    
    def inverse(self) -> 'RotationMatrix':
        """ Return the inverse of the rotation matrix.

        Returns
        -------
        RotationMatrix
            The inverse rotation matrix.
        """
        return RotationMatrix(self.T, dim=self.dim)
    
    inv = inverse
    
    def trace(self) -> torch.Tensor:
        """Return the trace of the [`RotationMatrix`][pymagbot.math.spatial.RotationMatrix].

        Returns
        -------
        torch.Tensor
            The trace of the matrix.
        """
        return self.data.trace()
    
    def angle(self) -> torch.Tensor:
        """Returns absolute deviation of the [`RotationMatrix`][pymagbot.math.spatial.RotationMatrix]
         from identity

        Returns
        -------
        torch.Tensor
            The rotation angle in radians.
        """
        cos_theta = (self.trace() - self.dim) / 2.0
        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)  # Ensure numerical stability
        return torch.acos(cos_theta)
    
    def deskew(self) -> Vertex:
        """Get the skew-symmetric matrix representation of the rotation.

        Extracts the rotation vector from a skew-symmetric cross-product matrix or,
        if the matrix is a full rotation matrix, returns the closest equivalent representation.
        
        If $_rotation$ is a (flattened) cross-product matrix of the vector $w$, this will
        return the vector $w$.  Otherwise, it will return a representation w of
        $\\frac{(_rotation-_rotation^T)}{2}$ (off diagonals of $_rotation$) such that
        $\\frac{(_rotation-_rotation^T)}{2} = \\text{crossproduct}(w)$.

        If $_rotation$ is a valid rotation matrix:
          - In 3D, it returns a vector $w$ such that $\\mathrm{crossproduct}(w) \\approx (_rotation - _rotation^T)/2$$.
          - In 2D, it returns a single-element vector representing the angle.
          
        Returns
        -------
        Vertex
            A 3D vector $w$ such that $\\mathrm{cross_product}(w) \\approx (_rotation - _rotation^T)/2$$
        """
        if self.dim == 3:
            # Compute the skew-symmetric part: (_rotation - _rotation^T) / 2
            skew_part = 0.5 * (self.data - self.data.T)
            return Vertex([skew_part[2, 1], skew_part[0, 2], skew_part[1, 0]])
        elif self.dim == 2:
            # The 'skew' part of a 2D rotation matrix gives sin(theta)
            return Vertex([torch.asin(self.data[1, 0])])  # Simplified for 2D
        else:
            raise ValueError(f"deskew not supported for dimension {self.dim}")
    
    def rotation_vector(self, degrees=False) -> 'Direction':
        """Convert the rotation matrix to a rotation vector (exponential map).
        
        A rotation vector is a vector which is co-directional to the axis of rotation and whose norm
        gives the angle of rotation.
        Specifically, the rotation vector $w$ of a rotation matrix $R$ is such that:
        $\\mathrm{exp}(w) = R$. This is equivalent to data-angle representation with
        $\\frac{w}{||w||}=\\mathrm{R}$ and ${||w||}=\\mathrm{angle}$
        
        Parameters
        ----------
        degrees: bool, optional
            If this flag is set to `True`, the returned magnitudes are in degrees, otherwise in radians.
            Default is `False`.
        Returns
        -------
        Direction
            A rotation vector (axis-angle representation). In 2D, it represents the angle, in 3D the axis-angle.
        """
        angle = self.angle()
        if self.dim == 3:
            if angle == 0:
                return Direction(torch.zeros(3, dtype=DTYPE, device=self.device))  # Handle identity matrix case
            axis = create_unit_vector(
                self.deskew().data)  # self.deskew().other / torch.sin(angle) #Create direction/ axis
            rot_vec = axis * angle
            return Direction(rot_vec, dim=self.dim)
        elif self.dim == 2:
            if degrees:
                angle = torch.rad2deg(angle)
            return Direction([angle])
        else:
            raise ValueError(f"rotation_vector is not supported for dimension {self.dim}")
    
    def axis(self) -> torch.Tensor:
        """Get the rotation axis of the matrix."""
        return self.rotation_vector().data
    
    def distance(self, other: Optional[Union[torch.Tensor, ArrayLike, 'RotationMatrix', list, tuple]]) -> torch.Tensor:
        """
        Compute the geodesic distance between two rotation matrices.

        Parameters
        ----------
        other : array_like
            Another rotation matrix.

        Returns
        -------
        torch.Tensor
            The geodesic distance.
        """
        other: RotationMatrix = self.check(other)
        r_diff: RotationMatrix = self.inverse() @ other
        return r_diff.angle()
    
    @staticmethod
    def from_rotation_vector(rot_vec: Union[ArrayLike, torch.Tensor, MatrixBase, list, tuple]) -> 'RotationMatrix':
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
        return RotationMatrix(vector_to_rotation(rot_vec))
    
    @staticmethod
    def from_quaternion(quaternion: Union[ArrayLike, torch.Tensor, MatrixBase, list, tuple]) -> 'RotationMatrix':
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
        - [1] <https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation>
        - [2] <https://en.wikipedia.org/wiki/Quaternion>
        """
        return RotationMatrix(quaternion_to_rotation(quaternion))
    
    def to_quaternion(self) -> torch.Tensor:
        """Convert the [`RotationMatrix`][pymagbot.math.spatial.RotationMatrix] to a quaternion (x, y, z, w)."""
        return torch.tensor(rotation_to_quaternion(self.data), dtype=DTYPE, device=self.device)
    
    @staticmethod
    def from_euler(angles: Union[ArrayLike, torch.Tensor, MatrixBase, list, tuple], order: str = 'xyz',
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
        return RotationMatrix(euler_to_rotation(angles, order=order, degrees=degrees))
    
    def to_euler(self, order: str = 'xyz', degrees: bool = False) -> torch.Tensor:
        """Convert the [`RotationMatrix`][pymagbot.math.spatial.RotationMatrix] object to Euler angles.
        For docstring and variables see also [to_euler() in _transform.py]"""
        return torch.tensor(rotation_to_euler(self.data, order=order, degrees=degrees), dtype=DTYPE, device=self.device)


class RigidTransform(TransformBase):
    """A rigid transformation (SE(2) or SE(3)) between two  Euclidean spaces.

    **Rigid transformations** (or Euclidean transformation) include rotations, translations or any sequence of these.
    Any proper rigid transformation can be decomposed into a rotation followed by a translation.
    The set of proper rigid transformations is called _special Euclidean group_, denoted $SE(n)$ (e.g. n=2 or 3).
    In `pymagbot`, any primitive is defined in a 2D or 3D space.
    
    Attributes
    ----------
    rotation : RotationMatrix
        The rotation matrix.
    translation : Vertex
        The translation vector.
    dim : int
        The dimension of the rotation matrix (2 or 3).
        
    See also
    --------
    - [`RotationMatrix`][pymagbot.math.spatial.RotationMatrix] class for details about the rotation.
    
    References
    ----------
    - <https://en.wikipedia.org/wiki/Rigid_transformation>
    """
    
    def __init__(self, rotation: Optional[Union[ArrayLike, torch.Tensor, RotationMatrix, list, tuple]] = None,
                 translation: Optional[Union[ArrayLike, torch.Tensor, Vertex, list, tuple]] = None, dim: int = None):
        """
        Initialize the [`RigidTransform`][pymagbot.math.spatial.RigidTransform].

        Parameters
        ----------
        rotation : array_like, optional
            The rotation matrix.
        translation : array_like, optional
            The translation vector.
        dim : int, optional
            The dimension of the rigid transformation (2 or 3). Default 3.
        """
        self.device = DEVICE
        self._rotation = RotationMatrix(rotation, dim=dim)
        self._translation = Vertex(translation, dim=dim)
        self.dim = dim if dim is not None else self._translation.dim
        super().__init__(self.as_matrix(), dim=dim)  # data contains the homogeneous matrix
    
    @property
    def rotation(self) -> RotationMatrix:
        """Returns [`RotationMatrix`][pymagbot.math.spatial.RotationMatrix] rotation matrix"""
        return self._rotation
    
    @rotation.setter
    def rotation(self, r):
        if not isinstance(r, RotationMatrix):
            r = RotationMatrix(r)
        self._rotation = r
    
    @property
    def translation(self) -> Vertex:
        """Returns translation [`Vertex`][pymagbot.math.spatial.primitives.Vertex]"""
        return self._translation
    
    @translation.setter
    def translation(self, t):
        if not isinstance(t, Vertex):
            t = Vertex(t)
        self._translation = t
    
    def _initialize_data(self, data) -> torch.Tensor:
        self.dim: int = self.check_dim(data, self.dim)
        if self.dim > 3:
            self.dim -= 1
        return data
    
    def _validate(self):
        if self.data.shape != (self.dim + 1, self.dim + 1):
            raise ValueError(f"RigidTransform must be an homogeneous matrix with {self.dim + 1} dimensions"
                             f"the shape is {self.data.shape}")
    
    def __repr__(self) -> str:
        formatted_r = self.rotation._format_matrix("", prefix='RigidTransform(rotation=')
        formatted_t = self.translation._format_matrix("", prefix='RigidTransform(translation=')
        # return f"RigidTransform(rotation=\n{self._rotation!r},\ntranslation=\n{self.translation!r}\n)"
        return (f"RigidTransform(rotation={formatted_r},\n" +
                ' ' * len("RigidTransform(") + f"translation={formatted_t})")
    
    def __getitem__(self, idx):
        if idx == 0:
            return self.rotation
        elif idx == 1:
            return self.translation
        elif idx == (0, 1):
            return self.rotation, self.translation
        raise TypeError('Wrong index, only "0", "1", and "(0,1)" are supported')
    
    @classmethod
    def identity(cls, dim: int = None) -> 'RigidTransform':
        """
        Return the identity rigid transformation.

        Parameters
        ----------
        dim : int, optional
            The dimension of the rigid transformation (2 or 3). Default is 3.

        Returns
        -------
        RigidTransform
            The identity rigid transformation.
        """
        return cls(RotationMatrix.identity(dim), Vertex.zeros(dim))
    
    def inverse(self) -> 'RigidTransform':
        """
        Calculate the inverse of the rigid transformation.

        Returns
        -------
        RigidTransform
            The inverse rigid transformation.
        """
        r_inv = self.rotation.inverse()
        return RigidTransform(rotation=r_inv, translation=Vertex(-r_inv @ self.translation), dim=self.dim)
    
    def as_matrix(self) -> torch.Tensor:
        """
        Convert the rigid transformation to a homogeneous transformation matrix.

        Returns
        -------
        torch.Tensor
            The homogeneous transformation matrix (shape (dim+1, dim+1)).
        """
        # if self._data is not None:
        matrix = torch.eye(self.dim + 1, device=self.device, dtype=DTYPE)
        matrix[:self.dim, :self.dim] = self.rotation.data
        matrix[:self.dim, self.dim] = self.translation.data
        return matrix
    
    homogeneous = as_matrix
    
    @classmethod
    def from_matrix(cls, matrix: Optional[Union[torch.Tensor, ArrayLike, 'MatrixBase', list, tuple]]) -> \
            'RigidTransform':
        """
        Convert a  homogeneous transformation matrix to a rigid transformation.

        Returns
        -------
        RigidTransform
            The rigid transformation.
        """
        if isinstance(matrix, MatrixBase):
            matrix = matrix.data
        if torch.is_tensor(matrix):
            matrix = matrix.clone().to(DEVICE).float()  # Ensure it's a float tensor and on the correct device
        else:
            matrix = torch.tensor(matrix, dtype=DTYPE, device=DEVICE)  # Convert list/tuple to tensor
        dim = len(matrix) - 1
        return RigidTransform(rotation=matrix[:dim, :dim], translation=matrix[:dim, dim])
    
    @classmethod
    def from_homogeneous(cls, matrix: Optional[Union[torch.Tensor, ArrayLike, 'MatrixBase', list, tuple]]) -> \
            'RigidTransform':
        """
        Convert a  homogeneous transformation matrix to a rigid transformation.

        Returns
        -------
        RigidTransform
            The rigid transformation.
        """
        return cls.from_matrix(matrix)
    
    def to_dict(self) -> dict:
        """Convert the rigid transformation to a dictionary.

        Returns
        -------
        dict
            A dictionary representation of the rigid transformation.
        """
        return {'rotation': self.rotation.to_list(), 'translation': self.translation.to_list()}
    
    @classmethod
    def from_dict(cls, obj_dict: dict) -> 'RigidTransform':
        obj_dict = parse_dict(obj_dict)
        if not isinstance(obj_dict, dict) or 'rotation' not in obj_dict or 'translation' not in obj_dict:
            raise TypeError("obj_dict is not a valid `RigidTransform` dictionary structure")
        rotation = RotationMatrix(obj_dict['rotation'])
        translation: Vertex = Vertex(obj_dict['translation'])
        return cls(rotation=rotation, translation=translation, dim=translation.dim)
    
    def __matmul__(self, other):
        """
        Matrix multiplication between a [`RigidTransform`][pymagbot.math.spatial.RigidTransform] and an `other`
        objects.
        
        Note:
        -----
        The `other` object can be:
        
        - Another `RigidTransform`to produce another  `RigidTransform`.
        - A `Vertex` assumed to be a translation, then the transformation is applied on it.
        
        
        Parameters
        ----------
        other : array_like
            The other object to multiply.

        Returns
        -------
        RigidTransform or Vertex
            The result of the multiplication.
        Raises
        ------
        TypeError
            If the provided other type is not supported or if dimension mismatch.
        """
        if isinstance(other, RigidTransform):
            if self.dim != other.dim:
                raise ValueError(f"The dimension of the rigid _transform must match. Got {self.dim} and {other.dim}")
            return self.from_matrix(self.homogeneous() @ other.homogeneous())
        if isinstance(other, VectorBase):
            return self.apply_to(other.data)
        # Try to use MatrixBase multiplication (may raise an error)
        try:
            other = ensure_tensor(other)
            output: MatrixBase = super().__matmul__(other)
            return output
        except Exception as E:
            raise TypeError(f"Unsupported operation between RigidTransform and {type(other)}: {E}")
    
    def apply_to(self, point: Union[ArrayLike, torch.Tensor, Vertex, list, tuple]) -> Vertex:
        """Apply the rigid transformation to a point.

        Parameters
        ----------
        point : array_like
            The points to _transform.

        Returns
        -------
        Vertex
            The transformed points.
        """
        if not isinstance(point, Vertex):
            point = Vertex(point, dim=self.dim)
        if not isinstance(point, Vertex):
            raise TypeError("points must be a list, tuple, Vertex or an array_like")
        
        if point.dim == self.dim + 1:  # Homogeneous vector is provided
            return Vertex.from_homogeneous(self.as_matrix() @ point.data)
        elif point.dim == self.dim:
            return Vertex.from_homogeneous(self.as_matrix() @ point.homogeneous())
