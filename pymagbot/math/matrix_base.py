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
"""
This module defines base classes for matrices and vectors, providing a foundation for more specialized  geometric
representations like rotation matrices, points, and directions. It leverages  [PyTorch](https://pytorch.org/) for
efficient tensor operations and GPU acceleration if available.

!!! note
    This module provides base classes and should be extended to create more specific matrix and vector types
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import torch

from pymagbot._config import USE_CUDA

__all__ = ['MatrixBase', 'VectorBase']


class MatrixBase(ABC):
    """Abstract base class for matrix-like objects.

    This class provides a foundation for representing matrices using [PyTorch](https://pytorch.org/) tensors.
    It handles device management (CPU or GPU) and provides basic validation.

    Attributes
    ----------
    _data : torch.Tensor
        The data stored in the matrix.
        depending on the subclass. Can be a [PyTorch](https://pytorch.org/) tensor, another
        [`MatrixBase`][pymagbot.math.matrix_base.MatrixBase] instance, or a list/tuple of numerical values.
    dim : int, optional
        The dimension of the matrix (e.g. 2 for 2D, 3 for 3D).
    device : torch.device, optional
        The device on which the [`MatrixBase`][pymagbot.math.matrix_base.MatrixBase] instance is build
        (e.g. "cpu" or "cuda").
    """
    
    def __init__(self, data: Optional[Union[torch.Tensor, 'MatrixBase', list, tuple]] = None, dim: int = 3):
        """
        Initialize the [`MatrixBase`][pymagbot.math.matrix_base.MatrixBase] with data or shape.

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
        Get the underlying [PyTorch](https://pytorch.org/) tensor data.

        Returns
        -------
        torch.Tensor
            The [PyTorch](https://pytorch.org/) tensor representing the matrix data.
        """
        return self._data
    
    @data.setter
    def data(self, value: Union[torch.Tensor, 'MatrixBase', list, tuple]):
        """
        Set the underlying [PyTorch](https://pytorch.org/) tensor data.

        Parameters
        ----------
        value : Union[torch.Tensor, [`MatrixBase`][pymagbot.math.matrix_base.MatrixBase],, list, tuple]
            The new data for the matrix.  It will be converted to a [PyTorch](https://pytorch.org/) tensor and moved to the appropriate device.
        """
        self._data = self._initialize_data(value)
        self._validate()
    
    def _initialize_data(self, data) -> torch.Tensor:
        """
        Initialize the [PyTorch](https://pytorch.org/) tensor data from various input types.

        Parameters
        ----------
        data : Union[torch.Tensor, 'MatrixBase', list, tuple], optional
            The input data.

        Returns
        -------
        torch.Tensor
            The initialized [PyTorch](https://pytorch.org/) tensor.
        """
        if data is None:
            # shape = (self.dim, self.dim) if isinstance(self, RotationMatrix) else (self.dim,)
            return torch.zeros((self.dim,), device=self.device, dtype=torch.float32)
        
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
        return str(self.array())
    
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
    
    def __add__(self, other):
        """
        Add two [`MatrixBase`][pymagbot.math.matrix_base.MatrixBase] objects.

        Parameters
        ----------
        other : MatrixBase
            The other [`MatrixBase`][pymagbot.math.matrix_base.MatrixBase] object to add.

        Returns
        -------
        MatrixBase
            The result of the addition.
        """
        return self.__class__(self.data + other.data)
    
    def __sub__(self, other):
        """
        Subtract two [`MatrixBase`][pymagbot.math.matrix_base.MatrixBase] objects.

        Parameters
        ----------
        other : MatrixBase
            The other [`MatrixBase`][pymagbot.math.matrix_base.MatrixBase] object to subtract.

        Returns
        -------
        MatrixBase
            The result of the subtraction.
        """
        return self.__class__(self.data - other.data)
    
    def __matmul__(self, other):
        """
        Matrix multiplication of two [`MatrixBase`][pymagbot.math.matrix_base.MatrixBase] objects.

        Parameters
        ----------
        other : MatrixBase
            The other [`MatrixBase`][pymagbot.math.matrix_base.MatrixBase] object to multiply.

        Returns
        -------
        MatrixBase
            The result of the [`MatrixBase`][pymagbot.math.matrix_base.MatrixBase] multiplication.
        """
        return self.__class__(self.data @ other.data)
    
    def __mul__(self, scalar):
        """
        Multiply the [`MatrixBase`][pymagbot.math.matrix_base.MatrixBase] by a scalar.

        Parameters
        ----------
        scalar : float
            The scalar to multiply.

        Returns
        -------
        MatrixBase
            The result of the scalar multiplication.
        """
        return self.__class__(self.data * scalar)
    
    def __truediv__(self, scalar):
        """
        Divide the [`MatrixBase`][pymagbot.math.matrix_base.MatrixBase] by a scalar.

        Parameters
        ----------
        scalar : float
            The scalar to divide.

        Returns
        -------
        MatrixBase
            The result of the scalar division.
        """
        return self.__class__(self.data / scalar)


class VectorBase(MatrixBase):
    """Base class for vector-like objects, such as  [`Point`][pymagbot.math.geometry_base.Point]
    and [`Direction`][pymagbot.math.geometry_base.Direction].
    """
    
    def _validate(self):
        if self.data.shape != (self.dim,):
            raise ValueError(f"Vector must have shape ({self.dim},), got {self.data.shape}.")
