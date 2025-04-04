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
""" operations - spatial operations module

This module provides functions for converting between different representations of rotations,
including rotation vectors, quaternions, and Euler angles. It also includes functions for
creating rotation matrices from these representations and extracting angles from rotation matrices.

"""
from typing import Union

import numpy as np
import torch
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation

from pymagbot.utils.config import DTYPE, USE_CUDA
from pymagbot.math.matrix_base import MatrixBase

__all__ = ['vector_to_rotation', 'quaternion_to_rotation', 'euler_to_rotation',
           'rotation_to_quaternion', 'rotation_to_euler', 'rotation_to_angle_2d']


def check_as_tensor(x: Union[ArrayLike, torch.Tensor, MatrixBase, list, tuple], dim: int = None) -> torch.Tensor:
    """Helper to convert input to  (`torch`) tensor.
    
    Parameters
    ----------
    x : Union[ArrayLike, torch.Tensor, MatrixBase, list, tuple]
        The input other.
    dim : int, optional
        The dimension of the tensor, by default None.

    Returns
    -------
    torch.Tensor
        The tensor representation of the input.

    Raises
    ------
    ValueError
        If the shape of the input is invalid.
    """
    if isinstance(x, MatrixBase):
        x = x.data
    elif not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x, dtype=DTYPE, device=torch.device("cuda" if USE_CUDA else "cpu"))
    if dim is None:
        dim = x.shape[0] if len(x.shape) == 1 else x.shape[0]
    if len(x.shape) == 2 and x.shape != (dim, dim):
        raise ValueError(f"Invalid {dim}D rotation matrix shape {x.shape}, expected {(dim, dim)}")
    elif len(x.shape) == 1 and x.shape[0] != dim:
        raise ValueError(f"Invalid {dim}D vector shape {x.shape}, expected {(dim,)}")
    return x


def check_as_array(x: Union[ArrayLike, torch.Tensor, MatrixBase, list, tuple], dim: int = None) -> ArrayLike:
    """Helper to convert input to (`numpy`) array.

    Parameters
    ----------
    x : Union[ArrayLike, torch.Tensor, MatrixBase, list, tuple]
        The input other.
    dim : int, optional
        The dimension of the array, by default None.

    Returns
    -------
    ArrayLike
        The numpy array representation of the input.

    Raises
    ------
    ValueError
        If the shape of the input is invalid.
    """
    if isinstance(x, MatrixBase):
        x = x.array
    elif isinstance(x, torch.Tensor):
        x = x.data.cpu().numpy()
    elif not isinstance(x, ArrayLike):
        x = np.array(x, dtype=DTYPE)
    if dim is None:
        dim = x.shape[0] if len(x.shape) == 1 else x.shape[0]
    
    if len(x.shape) == 2 and x.shape != (dim, dim):
        raise ValueError(f"Invalid {dim}D rotation matrix shape {x.shape}, expected {(dim, dim)}")
    elif len(x.shape) == 1 and x.shape[0] != dim:
        raise ValueError(f"Invalid {dim}D vector shape {x.shape}, expected {(dim,)}")
    return x


def vector_to_rotation(rot_vec: Union[ArrayLike, torch.Tensor, MatrixBase, list, tuple]) -> torch.Tensor:
    """
    Create a rotation matrix from a rotation vector (exponential map).

    Parameters
    ----------
    rot_vec : Union[ArrayLike, torch.Tensor, list, tuple]
        A rotation vector (axis-angle representation).

    Returns
    -------
    torch.Tensor
        The rotation matrix.
    """
    rot_vec = check_as_tensor(rot_vec)
    if rot_vec.shape[0] == 3:
        angle = torch.linalg.norm(rot_vec)
        if angle == 0:
            return torch.eye(3, dtype=DTYPE, device=rot_vec.device)  # Handle zero angle case
        
        axis = rot_vec / angle
        a = torch.cos(angle)
        b = torch.sin(angle)
        v = axis.reshape(3, 1)
        skew_sym = torch.tensor([[0, -v[2, 0], v[1, 0]],
                                 [v[2, 0], 0, -v[0, 0]],
                                 [-v[1, 0], v[0, 0], 0]], dtype=DTYPE, device=rot_vec.device)
        rot = a * torch.eye(3, dtype=DTYPE, device=rot_vec.device) + b * skew_sym + (1 - a) * (v @ v.T)
        return rot
    elif rot_vec.shape[0] == 1:
        # 2D rotation from angle
        angle = rot_vec[0]
        rot = torch.tensor([[torch.cos(angle), -torch.sin(angle)],
                            [torch.sin(angle), torch.cos(angle)]], dtype=DTYPE, device=rot_vec.device)
        return rot
    else:
        raise ValueError(f"Invalid rotation vector dimension: {rot_vec.shape[0]}")


def quaternion_to_rotation(quaternion: Union[ArrayLike, torch.Tensor, MatrixBase, list, tuple]) -> torch.Tensor:
    """
    Create a rotation matrix from quaternion representation.

    Parameters
    ----------
    quaternion : array_like
        Quaternion as [x, y, z, w].

    Returns
    -------
    torch.Tensor
        A tensor that contains the quaternion angle.
    References
    ----------
    - [1] https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
    - [2] https://en.wikipedia.org/wiki/Quaternion
    """
    quaternion = check_as_tensor(quaternion)
    x, y, z, w = quaternion
    Nq = w * w + x * x + y * y + z * z
    if Nq < torch.finfo(DTYPE).eps:
        return torch.eye(3, dtype=DTYPE, device=quaternion.device)
    
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
    return torch.tensor(
        [[1.0 - (yY + zZ), xY - wZ, xZ + wY],
         [xY + wZ, 1.0 - (xX + zZ), yZ - wX],
         [xZ - wY, yZ + wX, 1.0 - (xX + yY)]], dtype=DTYPE, device=quaternion.device
    )


def rotation_to_quaternion(rot: Union[ArrayLike, torch.Tensor, MatrixBase, list, tuple]) -> ArrayLike:
    """Convert a rotation matrix to quaternion.

    Parameters
    ----------
    rot : Union[ArrayLike, torch.Tensor, MatrixBase, list, tuple]
        The rotation matrix.

    Returns
    -------
    np.ndarray
        The quaternion representation."""
    rot = check_as_array(rot)
    return Rotation.from_matrix(matrix=rot).as_quat()


def euler_to_rotation(angles: Union[ArrayLike, torch.Tensor, MatrixBase, list, tuple], order: str = 'xyz',
                      degrees: bool = False) -> torch.Tensor:
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
    torch.Tensor
        The rotation matrix.
    """
    angles = check_as_tensor(angles)
    if degrees:
        angles = torch.deg2rad(angles)
    
    if order in ('z', 'y', 'x') and angles.shape[0] == 1 and angles.numel() == 1:
        if order == 'z':
            _rot = torch.tensor([[torch.cos(angles[0]), -torch.sin(angles[0])],
                                 [torch.sin(angles[0]), torch.cos(angles[0])]], dtype=DTYPE, device=angles.device)
            return _rot
        elif order == 'y':
            _rot = torch.tensor([[torch.cos(angles[0]), torch.sin(angles[0])],
                                 [-torch.sin(angles[0]), torch.cos(angles[0])]], dtype=DTYPE, device=angles.device)
            return _rot
        elif order == 'x':  # for similarity and consistency
            _rot = torch.tensor([[torch.cos(angles[0]), -torch.sin(angles[0])],
                                 [torch.sin(angles[0]), torch.cos(angles[0])]], dtype=DTYPE, device=angles.device)
            return _rot
    else:
        def _axis_angle_rot(axis: str, angle: torch.Tensor):
            if axis == 'x':
                m_rot = torch.tensor([[1, 0, 0],
                                     [0, torch.cos(angle), -torch.sin(angle)],
                                     [0, torch.sin(angle), torch.cos(angle)]],
                                    dtype=DTYPE, device=angles.device)
            elif axis == 'y':
                m_rot = torch.tensor([[torch.cos(angle), 0, torch.sin(angle)],
                                     [0, 1, 0],
                                     [-torch.sin(angle), 0, torch.cos(angle)]],
                                    dtype=DTYPE, device=angles.device)
            elif axis == 'z':
                m_rot = torch.tensor([[torch.cos(angle), -torch.sin(angle), 0],
                                     [torch.sin(angle), torch.cos(angle), 0],
                                     [0, 0, 1]],
                                    dtype=DTYPE, device=angles.device)
            else:
                raise ValueError(f'invalid rotation axis "{axis}"')
            return m_rot
        
        rot = torch.eye(3, dtype=DTYPE, device=angles.device)
        for i, a in enumerate(order.lower()):
            rot = rot @ _axis_angle_rot(a, angles[i])
        return rot


def rotation_to_euler(rot: Union[ArrayLike, torch.Tensor, MatrixBase, list, tuple], order: str = 'xyz',
                      degrees: bool = False) -> ArrayLike:
    """Convert a rotation matrix to quaternion
    
    Attributes:
    -----------
    rot: ArrayLike
        A rotation matrix
    order: string
        Specifies sequence of axes for rotations. Up to 3 characters belonging to the set {‘X’, ‘Y’,  ‘Z’}
        for intrinsic rotations, or {‘x’, ‘y’, ‘z’} for extrinsic rotations. Extrinsic and intrinsic rotations
        cannot be mixed in one function call.  Default is 'xyz'.
    degrees: boolean, optional
        Returned angles are in degrees if this flag is `True`, else they are in radians. Default is `False`.
    Return:
    -------
    ArrayLike
        The Euler angle
    """
    rot = check_as_array(rot)
    return Rotation.from_matrix(rot).as_euler(seq=order, degrees=degrees)


def rotation_to_angle_2d(rot:  Union[ArrayLike, torch.Tensor, MatrixBase, list, tuple]) -> torch.Tensor:
    """Get the angle of a 2D rotation matrix.

    Parameters
    ----------
    rot : Union[ArrayLike, torch.Tensor, MatrixBase, list, tuple]
        The 2D rotation matrix.

    Returns
    -------
    torch.Tensor
        The rotation angle in radians.
    """
    rot = check_as_tensor(rot, dim=2)
    return torch.atan2(rot[1, 0], rot[0, 0])
