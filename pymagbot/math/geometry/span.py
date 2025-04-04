from typing import Sequence, Union

import numpy as np
import torch
from numpy.typing import ArrayLike

from pymagbot.math.matrix_base import MatrixBase, parse_dict
from pymagbot.utils.config import DEVICE, DTYPE


class Span:
    """Represents a span of values for creating a mesh grid.

    Attributes:
    ----------
    start : float
        The starting of the span.
    stop : float
        The end of the span.
    num : int
        The number of samples in the span.
    """
    
    def __init__(self, start: float = 0.0, stop: float = 1.0, num_samples: int = 1):
        """Initialize a [`Span`][pymagbot.math.geometry.Span] object.

        Parameters
        ----------
        start : float
            The starting name of the span.
        stop : float
            The ending name of the span.
        num_samples : int
            The number of samples in the span.
        """
        self.start: float = start
        self.stop: float = stop
        self.num: int = num_samples
    
    def __repr__(self) -> str:
        """String representation of the [`Span`][pymagbot.math.geometry.Span] object."""
        return f"Span(start={self.start}, stop={self.stop}, num_samples={self.num})"
    
    def __contains__(self, item: Union[tuple, list, int, float, torch.Tensor, ArrayLike, MatrixBase]):
        """Check if an item (e.g. value) is within the span.

        Parameters
        ----------
        item : array_like
            The item to check.

        Returns
        -------
        bool
            `True` if the item is within the span, `False` otherwise.
            Notes: if item contains several value, the item is considered within the span if **all values** are within
            the span.

        Raises
        ------
        TypeError
            If the item type is unsupported.
        """
        if isinstance(item, (tuple, list, int, float, ArrayLike)):
            item = torch.as_tensor(item, device=DEVICE, dtype=DTYPE)
        elif isinstance(item, MatrixBase):
            item = item.data
        if not isinstance(item, torch.Tensor):
            raise TypeError('Unsupported `item` type.')
        return torch.logical_and(torch.all(self.start >= item), torch.all(item <= self.stop))
    
    def __getitem__(self, idx):
        if idx == 0:
            return self.start
        elif idx == 1:
            return self.stop
        elif idx == 2:
            return self.num
        else:
            raise IndexError("Index out of range. Span only has 3 attributes: start, stop, and num_samples.")
    
    def __len__(self) -> int:
        """Get the number of samples in the span."""
        return self.num
    
    def __iter__(self):
        """Iterate over the span."""
        for i in range(self.num):
            yield self.start + i * self.step
    
    def __eq__(self, other: 'Span') -> bool:
        """Check if two [`Span`][pymagbot.math.geometry.Span] objects are equal."""
        return (self.start == other.start) and (self.stop == other.stop) and (self.num == other.num)
    
    def __ne__(self, other: 'Span') -> bool:
        """Check if two [`Span`][pymagbot.math.geometry.Span] objects are not equal."""
        return not self.__eq__(other)
    
    def __add__(self, other) -> 'Span':
        """Add `other` to` [`Span`][pymagbot.math.geometry.Span]."""
        if isinstance(other, (int, float)):
            return Span(self.start + other, self.stop + other, self.num)
        return Span(self.start + other.start, self.stop + other.stop, self.num )
    
    def __sub__(self, other) -> 'Span':
        """Subtract  `other` to [`Span`][pymagbot.math.geometry.Span]."""
        if isinstance(other, (int, float)):
            return Span(self.start - other, self.stop - other, self.num)
        return Span(self.start - other.start, self.stop - other.stop, self.num )
    
    def __mul__(self, other) -> 'Span':
        """Multiply other [`Span`][pymagbot.math.geometry.Span] objects."""
        if isinstance(other, (int, float)):
            return Span(self.start * other, self.stop * other, self.num)
        return Span(self.start * other.start, self.stop * other.stop, self.num )
    
    def scale(self, factor: float):
        """Scale the [`Span`][pymagbot.math.geometry.Span] by a factor.

        Parameters
        ----------
        factor : float
            The scaling factor.

        Returns
        -------
        Span
            The scaled [`Span`][pymagbot.math.geometry.Span] object.
        """
        self.start *= factor
        self.stop *= factor
    
    def logspace(self, base: float = 10.0) -> torch.Tensor:
        """ Generate evenly on a log scale within the span using[`torch.logspace`][torch.logspace]

        Returns
        -------
        torch.Tensor
            A tensor of evenly spaced points within the span.
        """
        return torch.logspace(self.start, self.stop, self.num, base=base, device=DEVICE, dtype=DTYPE)
    
    def linspace(self) -> torch.Tensor:
        """Generate evenly spaced points within the span using [`torch.linspace`][torch.linspace].

        Returns
        -------
        torch.Tensor
            A tensor of evenly spaced points within the span.
        """
        return torch.linspace(self.start, self.stop, self.num, device=DEVICE, dtype=DTYPE)
    
    to_tensor = linspace
    """Convert the span to a [PyTorch](https://pytorch.org/) tensor."""
    
    def to_array(self) -> np.ndarray:
        """Convert the span to a [NumPy](https://numpy.org/) array.

        Returns
        -------
        np.ndarray
            An array containing the values in the span.
        """
        # return self.to_tensor().numpy()
        return np.linspace(self.start, self.stop, self.num)
    
    def to_dict(self) -> dict:
        """Converts the [`Span`][pymagbot.math.geometry.Span] object to a dictionary.

        Returns
        -------
        dict
            The object structure as a dictionary.
        """
        return {"start": self.start, "stop": self.stop, "num_samples": self.num}
    
    @classmethod
    def from_dict(cls, dict_obj: Union[dict, str]) -> 'Span':
        """Create a [`Span`][pymagbot.math.geometry.Span] object from a dictionary.

        Parameters
        ----------
        dict_obj : Union[dict, str]
            The imported dictionary.

        Returns
        -------
        Span
            The [`Span`][pymagbot.math.geometry.Span] object.

        Raises
        ------
        TypeError
            If the `dict_obj` is not a Span type.
        """
        dict_obj = parse_dict(dict_obj)
        start = dict_obj.get('start', 0)
        stop = dict_obj.get('stop', 0)
        num_samples = dict_obj.get('num_samples', 0)
        return cls(start=start, stop=stop, num_samples=num_samples)
    
    @property
    def step(self) -> float:
        """Spacing between values. """
        if self.num == 0:
            return 0.0
        return float((self.stop - self.start) / self.num)
    
    @step.setter
    def step(self, step):
        """Set number of samples to step. """
        self.num = int((self.stop - self.start) / step)
    
    @property
    def range(self) -> float:
        """Returns the `Span`'s range"""
        return self.stop - self.start
    
    @classmethod
    def from_tensor(cls, tensor: Union[torch.Tensor, MatrixBase, list, tuple, ArrayLike]) -> Sequence['Span']:
        """Create a [`Span`][pymagbot.math.geometry.Span] object from a tensor or array-like object.
        
        Assume that the tensor contains at least two values, then extract the start, stop, and num_samples from the
        min, max, and length of the tensor.
        
        Parameters
        ----------
        tensor : array_like
            A tensor containing the data to create spans from, shape: (N, dim).
        Returns
        -------
        Span
            The sequence of [`Span`][pymagbot.math.geometry.Span] object.
        """""
        if isinstance(tensor, MatrixBase):
            tensor = tensor.array
        elif not isinstance(tensor, torch.Tensor):
            tensor = torch.as_tensor(tensor, device=DEVICE, dtype=DTYPE)
        spans = []
        for col in tensor.T:
            start = torch.min(col).item()
            end = torch.max(col).item()
            num = len(col)
            spans.append(cls(start=start, stop=end, num_samples=num))
        return spans
    
    @classmethod
    def from_range(cls, num_samples: int, range: float) -> 'Span':
        """Create a Span object from a number of samples and a range.

        Parameters
        ----------
        num_samples : int
            The number of samples.
        range : float
            The range of the span.

        Returns
        -------
        Span
            The Span object.
        """
        return cls(start=0, stop=range, num_samples=num_samples)
    
    @classmethod
    def from_arange(cls, start: float, stop: float, step: float) -> 'Span':
        """Create a Span object from an [arange][numpy.arange].

        Parameters
        ----------
        start : float
            The starting name of the arange.
        stop : float
            The ending name of the arange.
        step : float
            The step size of the arange.

        Returns
        -------
        Span
            The Span object.
        """
        num_samples = int((stop - start) / step) + 1
        return cls(start=start, stop=stop, num_samples=num_samples)

    def normalize(self, points: torch.Tensor) -> torch.Tensor:
        """Normalize points to the range of [-1, 1] based on the span.

        Parameters
        ----------
        points : torch.Tensor
            Points to normalize, shape: (N).

        Returns
        -------
        torch.Tensor
            Normalized points, shape: (N).
        """
        return 2 * (points - self.start) / (self.stop - self.start) - 1