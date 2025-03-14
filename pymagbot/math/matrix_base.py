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
This module defines base classes for matrices and vectors, providing a foundation for more specialized  geometric
representations like rotation matrices, points, and directions. It leverages  [PyTorch](https://pytorch.org/) for
efficient tensor operations and GPU acceleration if available.

!!! note
    This module provides base classes and should be extended to create more specific matrix and vector types
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from numpy.typing import ArrayLike

from pymagbot._config import USE_CUDA

__all__ = ['MatrixBase', 'VectorBase']


def _get_data(data):
    if isinstance(data, MatrixBase):
        return data.data


class MatrixBase(ABC):
    """Abstract base class for matrix-like objects.

    This class provides a foundation for representing matrices using [PyTorch](https://pytorch.org/) tensors.
    It handles device management (CPU or GPU) and provides basic validation.

    Attributes
    ----------
    _data : torch.Tensor
        The data stored in the matrix.
        The data strongly depends on the subclass. Can be a [PyTorch](https://pytorch.org/) tensor,
        [Numpy](http://numpy.org/) array, another  [`MatrixBase`][pymagbot.math.matrix_base.MatrixBase] instance,
         or a [`list`][list]/[`tuple`][tuple] of numerical values.
    dim : int, optional
        The dimension of the matrix (e.g. 2 for 2D, 3 for 3D).
    device : torch.device, optional
        The device on which the [`MatrixBase`][pymagbot.math.matrix_base.MatrixBase] instance is build
        (e.g. "cpu" or "cuda").
    """
    
    dim: Optional[int] = 3
    
    def __init__(self, data: Optional[Union[torch.Tensor, ArrayLike, 'MatrixBase', list, tuple]] = None,
                 dim: int = 3):
        """
        Initialize the [`MatrixBase`][pymagbot.math.matrix_base.MatrixBase] with data or dim.

        Parameters
        ----------
        data : array_like, optional
            The data to initialize the matrix.
        dim : int, optional
            The dimension of the matrix. Only used if data is not provided.
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
    
    tensor = data
    
    @property
    def array(self) -> np.ndarray:
        """The data as [numpy](https://numpy.org/) array."""
        return self.data.cpu().numpy()
    
    @array.setter
    def array(self, value: Union[torch.Tensor, 'MatrixBase', list, tuple]):
        """
        Set the underlying [PyTorch](https://pytorch.org/) tensor data.

        Parameters
        ----------
        value : Union[torch.Tensor, [`MatrixBase`][pymagbot.math.matrix_base.MatrixBase],, list, tuple]
            The new data for the matrix.  It will be converted to a [PyTorch](https://pytorch.org/) tensor and moved to the appropriate device.
        """
        self.data = value
    
    matrix = array
    
    @classmethod
    def null(cls):
        """`null` value"""
        return cls(torch.empty(0))
    
    def isnull(self) -> bool:
        return self == self.null()
    
    @property
    def shape(self):
        """Shape of the [`MatrixBase`][pymagbot.math.matrix_base.MatrixBase] object."""
        if isinstance(self.data, torch.Tensor):
            return self.data.shape
        raise TypeError(f'Class {type(self)} does not support shape.')
    
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
            self.dim: int = data.shape[0]
            return data.clone().to(self.device).float()  # Ensure it's a float tensor and on the correct device
        data = torch.tensor(data, dtype=torch.float32, device=self.device)  # Convert list/tuple to tensor
        self.dim: int = data.shape[0]
        return data
    
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
    
    def __str__(self) -> str:
        """
        String representation of the matrix.

        Returns
        -------
        str
            A string representation of the matrix data (CPU numpy array).
        """
        return str(self.array)
    
    def __repr__(self) -> str:
        """
        String representation of the object.

        Returns
        -------
        str
            A string representation of the object including class name and data.
        """
        return f"{self.__class__.__name__}({self.array})"
    
    def to_list(self) -> List[Any]:
        """
        Convert the matrix data to a list.

        Returns
        -------
        list
            A list representation of the matrix data.
        """
        return self.array.tolist()
    
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
        Create an instance of a [`MatrixBase`][pymagbot.math.matrix_base.MatrixBase] from a dictionary.

        Parameters
        ----------
        data_dict : dict
            A dictionary containing the matrix data. Must have a "data" key.

        Returns
        -------
        MatrixBase
            A new instance of the  [`MatrixBase`][pymagbot.math.matrix_base.MatrixBase] initialized with the data
            from the dictionary.
        """
        return cls(data_dict["data"])
    
    def clone(self) -> 'MatrixBase':
        """
        Create a deep copy of the  [`MatrixBase`][pymagbot.math.matrix_base.MatrixBase] object.

        Returns
        -------
        MatrixBase
            A deep copy of the object.
        """
        return self.__class__(self.data.clone())
    
    def __getitem__(self, idx):
        return self._data[idx]
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __eq__(self, other) -> bool:
        if other is None:
            return False
        other = _get_data(other)
        if self.shape != other.shape:
            return False
        return torch.allclose(self.data, other)
    
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
        other = _get_data(other)
        return self.__class__(self.data + other)
    
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
        other = _get_data(other)
        return self.__class__(self.data - other)
    
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
    
    def norm(self, **kwargs):
        """ The  [`MatrixBase`][pymagbot.math.matrix_base.MatrixBase] norm of the matrix (if any)
        
        Parameters
        ----------
        **kwargs : dict
            Keyword arguments to be passed to the [torch.norm][] function.
        """
        return self.data.norm(**kwargs)
    
    def transpose(self, **kwargs):
        """Returns the   [`MatrixBase`][pymagbot.math.matrix_base.MatrixBase] with axes transposed."""
        return self.data.transpose(**kwargs)
    
    @property
    def T(self):
        """Returns the  [`MatrixBase`][pymagbot.math.matrix_base.MatrixBase] with axes transposed."""
        return self.transpose()
    
    def scale(self, factor):
        """Scale the [`MatrixBase`][pymagbot.math.matrix_base.MatrixBase] by a factor"""
        self.data.mul_(factor)
    
    def rmse(self, other):
        """Root-mean-square error between two [`MatrixBase`][pymagbot.math.matrix_base.MatrixBase]."""
        other = _get_data(other)
        return torch.sqrt(torch.mean((self.data - other) ** 2))


class VectorBase(MatrixBase):
    """Base class for vector-like objects, such as  [`Point`][pymagbot.math.geometry_base.Point]
    and [`Direction`][pymagbot.math.geometry_base.Direction].
    """
    
    def _validate(self):
        """
        Validate that the data has the correct shape for a [`VectorBase`][pymagbot.math.matrix_base.VectorBase].

        Raises
        ------
        ValueError
            If the data shape is not equal to `(self.dim,)`.
        """
        if self.data.shape != (self.dim,):
            raise ValueError(f"Vector must have shape ({self.dim},), got {self.data.shape}.")
    
    def _initialize_data(self, data) -> torch.Tensor:
        data = super()._initialize_data(data)
        self.dim = data.shape[0]
        return data
    
    @classmethod
    def null(cls):
        return cls(torch.zeros((cls.dim,)))
    
    @property
    def vector(self):
        """The  [`VectorBase`][pymagbot.math.matrix_base.VectorBase] components. """
        return self._data.squeeze()
    
    @property
    def x(self):
        """The first element in the [`VectorBase`][pymagbot.math.matrix_base.VectorBase]'s data."""
        return self.vector[0]
    
    @property
    def y(self):
        """The second element in the [`VectorBase`][pymagbot.math.matrix_base.VectorBase]'s data."""
        return self.vector[1]
    
    @property
    def z(self):
        """The third element in the  [`VectorBase`][pymagbot.math.matrix_base.VectorBase]'s data."""
        if self.dim < 3:
            raise (IndexError(f"z coordinates accessible in 3D!"))
        return self.vector[2]
