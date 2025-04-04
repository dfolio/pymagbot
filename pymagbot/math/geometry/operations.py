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
operations - Submodule for operation on spatial  primitives.

This module provides helper functions for coordinate system conversions and basic
vector operations. It is designed for use with `pymagbot` geometric primitives,
accepting and returning objects of type `Vertex` and `VectorBase`.

Supported conversions include:
- Cartesian to Cylindrical and Spherical (for coordinate)
- Cylindrical to Cartesian and Spherical
- Spherical to Cartesian and Cylindrical

Additionally, vector operations such as magnitude, dot product, and cross product
are provided.

Notes
-----
- In 2D, some functions will return scalar angles instead of multi-element vectors.
- The functions assume that input objects are instances of `Vertex` or `VectorBase`
  (defined\n  in `pymagbot/math/primitives.py`).
  
Examples
--------
```python
from pymagbot.math.geometry.operations import vertex_cartesian_to_cylindrical
from pymagbot.math.spatial.primitives import Vertex
v = Vertex([1.0, 1.0, 2.0])
rho, phi, z = vertex_cartesian_to_cylindrical(v)
print(rho, phi, z)
```
"""
import torch

from pymagbot.math.spatial.primitives import Vertex

__all__ = [
    'fix_nan',
    'vertex_cartesian_to_cylindrical', 'point_cartesian2spherical', 'point_cylindrical2cartesian',
    'point_cylindrical2spherical', 'point_spherical2cartesian', 'point_spherical2cylindrical',
    'vector_spherical2cartesian', 'vector_cylindrical2cartesian',
    'vector_magnitude', 'dot_product', 'cross_product']


def fix_nan(b):
    b[torch.isinf(b)] = torch.nan
    nans, x = torch.isnan(b), lambda z: z.nonzero()[0]
    b[nans] = torch.interp(x(nans), x(~nans), b[~nans])
    return b


def _check_vertex(vertex: Vertex, dim: int = 3) -> list:
    if vertex.dim != dim:
        raise ValueError(f"Vertex dim {vertex.dim} does not match the required {dim}")
    return vertex.to_list()


def vertex_cartesian_to_cylindrical(vertex: Vertex) -> Vertex:
    """Convert point Cartesian coordinates in cylindrical coordinates.
    
    Parameters
    ----------
    vertex : Vertex
        A vertex with Cartesian coordinates [x, y, z].

    Returns
    -------
    tuple
        A tuple (rho, phi, z), where:
            - rho is the radial distance,
            - phi is the azimuthal angle (in radians),
            - z is the height.
    """
    x, y, z = _check_vertex(vertex, dim=3)
    rho = vector_magnitude((x, y))
    phi = torch.arctan2(y, x)
    return Vertex([rho, phi, z])


def point_cartesian2spherical(x, y, z) -> tuple:
    """Convert point Cartesian coordinates in spherical coordinates."""
    r = vector_magnitude((x, y, z))
    with torch.errstate(divide='ignore', invalid='ignore'):
        theta = fix_nan(torch.arccos(x / r))
    phi = torch.arctan2(y, x)
    return r, theta, phi


def point_cylindrical2cartesian(rho, phi, z) -> tuple:
    """Convert point cylindrical coordinates in Cartesian coordinates."""
    x = rho * torch.cos(phi)
    y = rho * torch.sin(phi)
    return x, y, z


def point_cylindrical2spherical(rho, phi, z) -> tuple:
    """Convert point cylindrical coordinates in spherical coordinates."""
    r = vector_magnitude((rho, z))
    theta = torch.arctan2(rho, z)
    return r, theta, phi


def point_spherical2cartesian(r, theta, phi) -> tuple:
    """Convert point spherical coordinates in Cartesian coordinates."""
    x = r * torch.sin(theta) * torch.cos(phi)
    y = r * torch.sin(theta) * torch.sin(phi)
    z = r * torch.cos(theta)
    return x, y, z


def point_spherical2cylindrical(r, theta, phi) -> tuple:
    """Convert point spherical coordinates in cylindrical coordinates."""
    rho = r * torch.sin(theta)
    z = r * torch.sin(theta)
    return rho, phi, z


def vector_cartesian2cylindrical(v_x, v_y, v_z, x, y) -> tuple:
    """Convert vector Cartesian components in cylindrical components."""
    m = vector_magnitude((x, y))
    with torch.errstate(divide='ignore', invalid='ignore'):
        v_r = fix_nan((x * v_x + y * v_y) / m)
        v_phi = fix_nan((x * v_y - y * v_x) / m)
    return v_r, v_phi, v_z


def vector_cartesian2spherical(v_x, v_y, v_z, theta, phi) -> tuple:
    """Convert vector  Cartesian components in spherical components."""
    v_r = v_x * torch.sin(theta) * torch.cos(phi) + v_y * torch.sin(theta) * torch.sin(phi) + v_z * torch.cos(theta)
    v_theta = -v_x * torch.cos(theta) * torch.cos(phi) + v_y * torch.cos(theta) * torch.sin(phi) - v_z * torch.sin(
        theta)
    v_phi = -v_x * torch.sin(phi) + v_y * torch.cos(phi)
    return v_r, v_theta, v_phi


def vector_cylindrical2cartesian(v_r, v_phi, v_z, phi) -> tuple:
    """Convert vector  cylindrical components in Cartesian components."""
    v_x = v_r * torch.cos(phi) - v_phi * torch.sin(phi)
    v_y = v_r * torch.sin(phi) + v_phi * torch.cos(phi)
    return v_x, v_y, v_z


def vector_spherical2cartesian(v_r, v_theta, v_phi, phi, theta) -> tuple:
    """Convert vector spherical components in Cartesian components."""
    v_x = v_r * torch.sin(theta) * torch.cos(phi) + v_theta * torch.cos(theta) * torch.cos(phi) - v_phi * torch.sin(phi)
    v_y = v_r * torch.sin(theta) * torch.sin(phi) + v_theta * torch.cos(theta) * torch.sin(phi) + v_phi * torch.cos(phi)
    v_z = v_r * torch.cos(theta) + v_theta * torch.sin(theta)
    return v_x, v_y, v_z


def vector_magnitude(vector):
    """Get the magnitude of vector"""
    _magnitude = 0.
    for component in vector:
        _magnitude += component ** 2
    return _magnitude ** (1 / 2)


def dot_product(vector1, vector2) -> float:
    """Get the dot product between two components"""
    if not (4 > len(vector1) > 1):
        raise ValueError('Vectors should have at least 2 no more then 3 components.')
    if len(vector1) == 3:
        return vector1[0] * vector2[0] + vector1[1] * vector2[1] + vector1[2] * vector2[2]
    else:
        return vector1[0] * vector2[0] + vector1[1] * vector2[1]


def cross_product(vector1, vector2):
    """Cross product between two components"""
    if not (4 > len(vector1) > 1) and not (4 > len(vector2) > 1):
        raise ValueError('Vectors should have at least 2 no more then 3 components.')
    if len(vector1) != len(vector2):
        raise ValueError('Vectors should have the same number of components.')
    if len(vector1) == 2:
        return torch.as_tensor([vector1[0] * vector2[1] - vector1[1] * vector2[0]])
    if len(vector1) == 3:
        return torch.as_tensor([vector1[1] * vector2[2] - vector1[2] * vector2[1],
                                vector1[2] * vector2[0] - vector1[0] * vector2[2],
                                vector1[0] * vector2[1] - vector1[1] * vector2[0]])
