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
matrix_base - Submodule for matrix/tensor base objects.

This module defines [`MatrixBase`][pymagbot.math.matrix_base.MatrixBase], the core class for handling mathematical
operations on tensors. It supports operations on **scalars, vectors, and matrices**
while ensuring GPU acceleration with [**PyTorch**](https://pytorch.org/).

Notes
-----
This module provides base classes and should be extended to create more
specific matrix and vector types.
"""

import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

# import toml
import numpy as np
import torch
import yaml
from numpy.typing import ArrayLike

from pymagbot._config import DEVICE, DTYPE, FORMAT

__all__ = ['parse_dict', 'MatrixBase', 'ensure_tensor']


def parse_dict(dict_obj: Union[str, dict]) -> dict:
    """Convert an input object to a dictionary, supporting JSON and YAML formats.

    This function ensures that any input, whether a string (representing JSON or YAML)
    or a dictionary, is returned as a valid Python [`dict`][dict].

    Parameters
    ----------
    dict_obj : str or dict
        The input structure to validate and convert. If it is a string, the function attempts to parse it.

    Returns
    -------
    dict
        A valid Python dictionary parsed from the input.

    Raises
    ------
    TypeError
        If `dict_obj` is not a valid dictionary or cannot be parsed.
    ValueError
        If `dict_obj` is a string but not in a recognized format (JSON or YAML).

    Examples
    --------
    **Convert a JSON string to a dictionary:**
    ```python
    json_str = '{"key": "name", "num": 42}'
    result = to_dict(json_str)
    print(result)  # {'key': 'name', 'num': 42}
    ```

    **Convert a YAML string to a dictionary:**
    ```python
    yaml_str = "key: name\nnum: 42"
    result = to_dict(yaml_str)
    print(result)  # {'key': 'name', 'num': 42}
    ```
    """
    if isinstance(dict_obj, dict):
        return dict_obj
    
    if isinstance(dict_obj, str):
        # Try parsing JSON
        try:
            return json.loads(dict_obj)
        except json.JSONDecodeError:
            pass  # Continue to next format
        # Try parsing YAML
        try:
            return yaml.safe_load(dict_obj)
        except yaml.YAMLError:
            pass  # Continue to next format
        
        raise ValueError(f"`dict_obj` could not be parsed as JSON or YAML.")
    
    raise TypeError(f"`dict_obj` must be a string or dictionary, got {type(dict_obj).__name__}.")


def ensure_tensor(other, dtype=DTYPE, device: str = DEVICE) -> torch.Tensor:
    """Helper function to ensure that `other` is a [`torch.Tensor`][torch.Tensor]"""
    if isinstance(other, MatrixBase):
        return other.data
    if not isinstance(other, torch.Tensor):
        return torch.tensor(other, dtype=dtype, device=device)
    return other


class MatrixBase(ABC):
    """Abstract base class for matrix-like objects.

    This class provides a foundation for representing tensors and matrices using  [**PyTorch**](https://pytorch.org/)
    tensors. It ensures GPU compatibility, enables serialization, and defines basic mathematical operations.

    Attributes
    ----------
    _data : torch.Tensor
        Internal tensor representation of the matrix.
        The data shape strongly depends on the subclass.
    dim : int
        The dimension of the matrix (e.g. 2 for 2D, 3 for 3D).
    device : torch.device
        The computation device (`cpu` or `cuda`).
    """
    
    dim: Optional[int] = None
    _data: Optional[torch.Tensor] = None
    
    def __init__(self, data: Optional[Union[torch.Tensor, ArrayLike, 'MatrixBase', list, tuple]] = None,
                 dim: int = None):
        """
        Initialize the [`MatrixBase`][pymagbot.math.matrix_base.MatrixBase] with a tensor or dim.

        Parameters
        ----------
        data : array_like, optional
            The data to initialize the matrix.
        dim : int, optional
            The dimension of the matrix. Only used if data is not provided.
        """
        self.device = DEVICE
        self.dim: int = self.check_dim(data, dim)
        if self.dim not in (2, 3):
            raise ValueError(f"Only dim=2 (2D) and dim=3 (3D) is supported, and got {self.dim}")
        self._data = self._initialize_data(data)
        self._validate()
    
    def check_dim(self, data: Optional[Union[torch.Tensor, ArrayLike, 'MatrixBase', list, tuple]] = None,
                  dim: int = None) -> int:
        """Check/fix consistency of the `dim` attribute against the `data` name"""
        if self.dim is not None:
            return self.dim
        if data is None and dim is None:
            dim = 3  # Default dimension
        elif data is None and dim is not None:
            dim = dim
        else:  # if data is not None:
            if isinstance(data, MatrixBase):
                dim = data.data.shape[0] if len(data.data.shape) == 1 else data.data.shape[0]  # data.shape[0]
            elif torch.is_tensor(data):
                dim = data.shape[0] if len(data.shape) == 1 else data.shape[0]
            else:
                dim = len(data)
        return dim
    
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
    def data(self, value: Union[torch.Tensor, ArrayLike, 'MatrixBase', list, tuple]):
        """
        Set the underlying [PyTorch](https://pytorch.org/) tensor data.

        Parameters
        ----------
        value : array_like
            The new data for the matrix.  It will be converted to a [PyTorch](https://pytorch.org/) tensor and moved to the appropriate device.
        """
        self.dim: int = self.check_dim(value, self.dim)
        self._data = self._initialize_data(value)
        self._validate()
    
    tensor = data
    matrix = data
    
    @property
    def array(self) -> np.ndarray:
        """The data as [numpy](https://numpy.org/) array."""
        return self.data.cpu().numpy()
    
    @array.setter
    def array(self, value: Union[torch.Tensor, ArrayLike, 'MatrixBase', list, tuple]):
        """Set the underlying [PyTorch](https://pytorch.org/) tensor from an array_like object..

        Parameters
        ----------
        value : array_like
            The new data for the matrix.
        """
        self.data = value
    
    @classmethod
    def null(cls):
        """`null` name"""
        return cls(torch.empty(0))
    
    def isnull(self) -> bool:
        """Check if the MatrixBase is Null.
        Returns
        -------
        bool
            `True` if the MatrixBase is null, `False` otherwise.
        """
        return self == self.null()
    
    @property
    def shape(self):
        """Shape of the [`MatrixBase`][pymagbot.math.matrix_base.MatrixBase] object."""
        return self.data.shape
    
    def _initialize_data(self, data) -> torch.Tensor:
        """Initialize the [PyTorch](https://pytorch.org/) tensor data from various input types.

        Parameters
        ----------
        data : array_like], optional
            The input data.

        Returns
        -------
        torch.Tensor
            The initialized [PyTorch](https://pytorch.org/) tensor.
        """
        if data is None:
            # self.dim = self.dim if self.dim is not None else 3
            return torch.zeros((self.dim,), device=self.device, dtype=DTYPE)
        if isinstance(data, MatrixBase):
            data = data.data.clone()
        if torch.is_tensor(data):
            return data.to(self.device).float()  # Ensure it's a float tensor and on the correct device
        return torch.tensor(data, dtype=DTYPE, device=self.device)  # Convert list/tuple to tensor
    
    @abstractmethod
    def _validate(self):
        """Validate the shape and properties of the data.

        This method must be implemented by subclasses to ensure data consistency.

        Raises
        ------
        ValueError
            If the data does not meet the required criteria (e.g., shape mismatch, non-orthogonal matrix).
        """
        pass
    
    def __str__(self, indent: int = 0) -> str:
        """
        String representation of the matrix.

        Returns
        -------
        str
            A string representation of the matrix data (CPU numpy array).
        """
        return ' ' * indent + str(self.array)
    
    def __repr__(self) -> str:
        """String representation of the object.

        Returns
        -------
        str
            A string representation of the object including class name and data.
        """
        return self._format_matrix(self.__class__.__name__)
    
    def _format_matrix(self, classname: str, prefix: str = None) -> str:
        """Helper method to format the matrix with proper indentation and precision.

        Parameters
        ----------
        classname : str
            The class name to be included in the representation.
        prefix : str
            The class name to be included in the representation.

        Returns
        -------
        str
            The formatted matrix representation.
        """
        prefix = prefix if not None else classname
        # Retrieve formatting settings
        precision = FORMAT.get("precision", 3)
        sci_mode = FORMAT.get("sci_mode", False)
        align_columns = FORMAT.get("align_columns", True)
        max_width = FORMAT.get("max_width", 80)
        # Format the matrix string
        formatted_matrix = np.array2string(
            self.array,
            separator=', ',
            precision=precision,
            suppress_small=not sci_mode,  # Suppress scientific notation if sci_mode is False
            threshold=np.prod(self.array.shape),  # Ensure full matrix is displayed
            max_line_width=max_width,  # Prevent excessive line wrapping
            formatter={'float': lambda x: f"{x:.{precision}f}"} if not sci_mode else None
        )
        # Ensure column alignment
        if align_columns:
            lines = formatted_matrix.split("\n")
            formatted_matrix = "\n".join([" " * len(classname) + " " + line for line in lines])
        
        return f"{classname}({formatted_matrix})"
    
    def to_list(self) -> List[Any]:
        """Convert the array data to a list.

        Returns
        -------
        list
            A list representation of the matrix data.
        """
        return self.array.tolist()
    
    def to_dict(self) -> Dict[str, List[Any]]:
        """Serializes the matrix to a dictionary.

        Returns
        -------
        dict
            A dictionary representation of the matrix data.
        """
        return {"data": self.to_list()}
    
    @classmethod
    def from_dict(cls, data_dict: Dict[str, List[Any]]) -> 'MatrixBase':
        """Restore a matrix from a dictionary.

        Parameters
        ----------
        data_dict : dict
            A dictionary containing the matrix data.

        Returns
        -------
        MatrixBase
            A new instance initialized from the dictionary.
        """
        data_dict = parse_dict(data_dict)
        if not isinstance(data_dict, dict) or 'data' not in data_dict:
            raise ValueError(f"`data_dict` is not a valid {cls.__class__.__name__} dictionary.")
        return cls(data_dict["data"])
    
    @classmethod
    def from_str(cls, data_str: str) -> 'MatrixBase':
        """Restore a matrix from a string.

        Parameters
        ----------
        data_str : dict
            A dictionary containing the matrix data.

        Returns
        -------
        MatrixBase
            A new instance initialized from the dictionary.
        """
        return cls.from_dict(parse_dict(data_str))
    
    def to_json(self) -> str:
        """Serializes the matrix to a JSON string.

        Returns
        -------
        str
            JSON string representation.
        """
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str):
        """
        Loads a matrix from a JSON string.

        Parameters
        ----------
        json_str : str
            JSON-encoded matrix data.

        Returns
        -------
        MatrixBase
            An instance of the matrix class.
        """
        return cls.from_dict(parse_dict(json_str))
    
    def to_yaml(self):
        """Serializes the matrix to a YAML string.

        Returns
        -------
        str
            YAML string representation.
        """
        return yaml.dump(self.to_dict())
    
    @classmethod
    def from_yaml(cls, yaml_str):
        """Retrieve a matrix from a YAML string.

        Parameters
        ----------
        yaml_str : str
            YAML-encoded matrix data.

        Returns
        -------
        MatrixBase
            An instance of the matrix class.
        """
        return cls.from_dict(parse_dict(yaml_str))
    
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
    
    def __add__(self, other):
        """
        Add two [`MatrixBase`][pymagbot.math.matrix_base.MatrixBase] objects.

        Parameters
        ----------
        other : array_like
            The other data object to add.

        Returns
        -------
        MatrixBase
            The result of the addition.
        """
        return self.__class__(self.data + ensure_tensor(other, device=self.device))
    
    def __sub__(self, other):
        """
        Subtract two [`MatrixBase`][pymagbot.math.matrix_base.MatrixBase] objects.

        Parameters
        ----------
        other : array_like
            The other data object to subtract.

        Returns
        -------
        MatrixBase
            The result of the subtraction.
        """
        return self.__class__(self.data - ensure_tensor(other, device=self.device))
    
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
    
    def __matmul__(self, other):
        """Matrix multiplication of two [`MatrixBase`][pymagbot.math.matrix_base.MatrixBase] objects.

        Parameters
        ----------
        other : array_like
            The other data object to multiply.

        Returns
        -------
        MatrixBase
            The result of the [`MatrixBase`][pymagbot.math.matrix_base.MatrixBase] multiplication.
        """
        return self.__class__(self.data @ ensure_tensor(other, device=self.device))
    
    def __truediv__(self, other):
        """
        Divide the [`MatrixBase`][pymagbot.math.matrix_base.MatrixBase] by a other.

        Parameters
        ----------
        other : float
            The scalar to divide.

        Returns
        -------
        MatrixBase
            The result of the scalar division.
        """
        return self.__class__(self.data / ensure_tensor(other, device=self.device))
    
    def __floordiv__(self, other):
        return self.__class__(self._data // ensure_tensor(other, device=self.device))
    
    def __mod__(self, other):
        return self.__class__(self._data % ensure_tensor(other, device=self.device))
    
    def __pow__(self, other):
        return self.__class__(self._data ** ensure_tensor(other, device=self.device))
    
    def __neg__(self):
        return self.__class__(-self._data)
    
    def __pos__(self):
        return self.__class__(+self._data)
    
    def __abs__(self):
        return self.__class__(torch.abs(self._data))
    
    def __eq__(self, other) -> bool:
        """Check if two [`MatrixBase`][pymagbot.math.matrix_base.MatrixBase] are equal

        Parameters
        ----------
        other : array_like
            The second  [`MatrixBase`][pymagbot.math.matrix_base.MatrixBase]

        Returns
        -------
        bool
            `True` if all elements are equal.
        """
        if other is None:
            return False
        other = ensure_tensor(other, device=self.device)
        if self.shape != other.shape:
            return False
        return torch.equal(self.data, other)
    
    def __ne__(self, other):
        return not self.__eq__(ensure_tensor(other, device=self.device))
    
    def __lt__(self, other):
        return self._data < ensure_tensor(other, device=self.device)
    
    def __le__(self, other):
        return self._data <= ensure_tensor(other, device=self.device)
    
    def __gt__(self, other):
        return self._data > ensure_tensor(other, device=self.device)
    
    def __ge__(self, other):
        return self._data >= ensure_tensor(other, device=self.device)
    
    def __iadd__(self, other):
        self._data += ensure_tensor(other, device=self.device)
        return self
    
    def __isub__(self, other):
        self._data -= ensure_tensor(other, device=self.device)
        return self
    
    def __imul__(self, other):
        self._data *= ensure_tensor(other, device=self.device)
        return self
    
    def __imatmul__(self, other):
        self._data @= ensure_tensor(other, device=self.device)
        return self
    
    def __itruediv__(self, other):
        self._data /= ensure_tensor(other, device=self.device)
        return self
    
    def __ifloordiv__(self, other):
        self._data //= ensure_tensor(other, device=self.device)
        return self
    
    def __imod__(self, other):
        self._data %= ensure_tensor(other, device=self.device)
        return self
    
    def __ipow__(self, other):
        self._data **= ensure_tensor(other, device=self.device)
        return self
    
    def norm(self, **kwargs):
        """ The  [`MatrixBase`][pymagbot.math.matrix_base.MatrixBase] norm of the matrix (if any)
        
        Parameters
        ----------
        **kwargs : dict
            Keyword arguments to be passed to the [torch.norm][] function.
        """
        return self.data.norm(**kwargs)
    
    def transpose(self):
        """Returns the [`MatrixBase`][pymagbot.math.matrix_base.MatrixBase] with axes transposed."""
        # return self.data.transpose(0, -1)
        return self.T
    
    @property
    def T(self):
        """Returns the  [`MatrixBase`][pymagbot.math.matrix_base.MatrixBase] with axes transposed."""
        return self.__class__(self._data.T)
    
    def scale(self, factor: (int, float)):
        """Scale the [`MatrixBase`][pymagbot.math.matrix_base.MatrixBase] by a factor"""
        self.data.mul_(factor)
    
    def rmse(self, other):
        """Root-mean-square error between two [`MatrixBase`][pymagbot.math.matrix_base.MatrixBase]."""
        return torch.sqrt(torch.mean((self.data - ensure_tensor(other, device=self.device)) ** 2))
