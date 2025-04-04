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
frame - Submodule for frame objects.

This module defines the `Frame` class, which represents an abstract coordinate system in 2D (SE(2)) or 3D (SE(3)).
A frame is defined by a rigid transformation (rotation and translation) with respect to another reference frame.

Example
-------

```python
from pymagbot.math.frame import Frame
from pymagbot.math.spatial import RigidTransform
T = RigidTransform.identity()
world_frame = Frame.world
print(world_frame)
child_frame = Frame(name='child', parent=world_frame, _transform=T)
print(child_frame)
# Move child_frame translation in the world frame
child_frame.global_translation=[1.0, 2.0, 3.0]
print(f"{child_frame.name} position in world frame: {child_frame._transform.translation}")
```
 """

import itertools
import json
import weakref
from typing import Optional, Union

import torch
from numpy.typing import ArrayLike

from pymagbot.math.matrix_base import parse_dict
from pymagbot.math.spatial.transform import RigidTransform, RotationMatrix
from pymagbot.math.spatial.primitives import Direction, Vertex

__all__ = ['Frame', 'get_frame']

known_frames: weakref.WeakValueDictionary[str, 'Frame'] = weakref.WeakValueDictionary()  # Prevent memory leaks
"""Dictionary to track known frames and ensuring uniqueness."""

WORLD_FRAME_NAMES: tuple[str, ...] = ('world', 'Frame_0', 'F0', 'workspace')
"""Known reserved name and aliases for world frame of reference

!!! danger
    TODO: WORLD_FRAME_NAMES should be deprecated in future.
"""


class Frame:
    """Represents an abstract coordinate system in 2D (SE(2)) or 3D (SE(3)).
    
    This class is used to represent and handle **frames of reference** (or reference frames) either in 2D or 3D space.
    In physics, a frame is an abstract coordinate system. This means it has a starting point
    (i.e. [`Vertex`][pymagbot.math.spatial.primitives.Vertex]) and [`Direction`][pymagbot.math.spatial.primitives.Direction]
    specified in physical space. Specifically, the [`Frame`][pymagbot.math.frame.Frame] class manages _moving frames_.
    This is a frame of reference that moves with the observer (i.e. the  [`parent`][pymagbot.math.frame.Frame(parent)])
    along a trajectory (i.e. a curve).
    
    Also, each [`Frame`][pymagbot.math.frame.Frame] has a **unique** (case-insensitive) name.
    This name is stored in a dictionary called [`known_frames`][pymagbot.math.frame.known_frames].
    There is a sole global [world frame][pymagbot.math.frame.Frame.world] which is the parent (or ancestor) of all the
    others frames..
    
    Attributes
    ----------
    name : str
        The unique name of the frame.
    dim : int
        The dimension of the frame (2 or 3).
    parent : Frame
        The parent frame of the frame. Default the world frame is used as the parent.
    children : set[Frame]
        The set of children of the frame.
    _transform : RigidTransform
        The rigid transformation of the frame with respect to the `parent` frame.
    
    """
    _id_counter = itertools.count(1)
    """Unique ID generator for frames."""
    
    world = None
    """The unique world (global, absolute) frame of reference."""
    
    def __new__(cls, name: Optional[str] = None, parent: Optional['Frame'] = None,
                transform: Optional[RigidTransform] = None, dim: int = 3) -> 'Frame':
        """
        Create a new  [`Frame`][pymagbot.math.frame.Frame] instance or retrieve an existing one with the same name.

        This method ensures that only one [`Frame`][pymagbot.math.frame.Frame] instance exists for each unique name.

        Parameters
        ----------
        name : str, optional
            The name of the frame. If `None`, a unique name is generated.
        parent : Frame, optional
            The parent frame. If `None`, the world frame is used as the parent.
        transform : RigidTransform, optional
            The rigid transformation of the frame with respect to the parent frame.
            If `None`, an identity transformation is used.
        dim : int, optional
            The dimension of the frame (2 or 3). Defaults to 3.
            
        Returns
        -------
        Frame
            The [`Frame`][pymagbot.math.frame.Frame] instance.  If a frame with the given name already exists,
            in  [`known_frames`][pymagbot.math.frame.known_frames] that instance is returned.
            
        """
        name = name.lower() if name else f"frame_{next(cls._id_counter)}"
        if name in known_frames:
            return known_frames[name]
        instance = super().__new__(cls)
        instance._name = name  # Store the name early
        known_frames[name] = instance
        return instance
    
    def __init__(self, name: Optional[str] = None, parent: Optional['Frame'] = None,
                 transform: Optional[RigidTransform] = None, dim: int = 3):
        """
        Initialize the [`Frame`][pymagbot.math.frame.Frame] instance.

        Parameters
        ----------
        name : str, optional
            The name of the frame. The name of a frame must be unique (case-insensitive).
            If `None`, a name is automatically generated.
        parent : Frame, optional
            The parent frame. If `None`, the [world][pymagbot.math.frame.Frame.world] frame is used as the parent.
        transform : RigidTransform, optional
            The rigid transformation of the frame with respect to the parent frame.
            If `None`, an identity transformation is used.
            
            !!! seealso
                [`RigidTransform`][pymagbot.math.spatial.RigidTransform] class for details about rigid
                transformation.
            
        dim : int, optional
            The dimension of the frame (e.g. 2 or 3).
        """
        
        if hasattr(self, 'initialized'):
            return  # Prevent double initialization
        
        self.name: str = self._name
        self.dim: int = dim
        self.children: set['Frame'] = set()
        self._transform: RigidTransform = RigidTransform(dim=dim) if transform is None else transform
        if self.is_world():
            if Frame.world is None:
                Frame.world = self  # Set the world frame
            elif Frame.world is not self:
                raise ValueError(f"Only one world frame instance is allowed: {self.name:}.")
        self.parent: 'Frame' = Frame.world if parent is None and not self.is_world() else parent
        if self.parent:
            self.parent.children.add(self)
        
        self.initialized: bool = True
    
    @property
    def name(self) -> str:
        return self._name
    
    @name.setter
    def name(self, name: str):
        """Set/Change the name of the frame."""
        name: str = name.lower()
        if name not in known_frames:
            known_frames.pop(self._name, None)  # Remove the old name
            self._name = name
            known_frames[self._name] = self  # Add the new name
    
    def __iter__(self):
        yield from self.children
    
    def detach(self):
        """Detach the frame from its parent but keep children intact."""
        if self.parent and not self.parent.is_world():
            self.parent.children.discard(self)
            self.parent = Frame.world
    
    def is_world(self) -> bool:
        """Check if the frame is the world frame.

        Returns
        -------
        bool
            `True` if the frame's name is in the list of world frame names, `False` otherwise.
        """
        return self.name in WORLD_FRAME_NAMES
    
    is_reference = is_world
    
    @property
    def local_transform(self) -> RigidTransform:
        """Get the local transform of the frame with respect to its parent."""
        return self._transform
    
    @local_transform.setter
    def local_transform(self, transform: Union[RigidTransform, ArrayLike, torch.Tensor, list, tuple]):
        """Set the local transform of the frame.

        Parameters
        ----------
        transform : array_like
            The new local transform. It will be converted to a RigidTransform.
        """
        if not isinstance(transform, RigidTransform):
            transform = RigidTransform(transform, dim=self.dim)
        self._transform = transform
        self._propagate_transformation()
    
    @property
    def transform_to_world(self) -> RigidTransform:
        """The rigid transformation of the frame with respect to the (global/absolute) world frame."""
        transform: RigidTransform = self.local_transform
        if self.parent is None or self.parent.is_world():
            return transform
        return transform @ self.parent.transform_to_world
    
    @property
    def local_translation(self) -> Vertex:
        return self.local_transform.translation
    
    @local_translation.setter
    def local_translation(self, translation: Union[ArrayLike, torch.Tensor, Vertex, list, tuple]):
        """Set the translation component of the frame's (local) parent transformation.

        Parameters
        ----------
        translation : array_like
            The translation vector with respect to the parent frame.
        """
        if not isinstance(translation, Vertex):
            translation = Vertex(translation, dim=self.dim)
        self._transform.translation = translation
        self._propagate_transformation()
    
    origin = local_translation
    
    @property
    def global_translation(self) -> Vertex:
        return self.transform_to_world.translation
    
    @global_translation.setter
    def global_translation(self, translation: Union[ArrayLike, torch.Tensor, Vertex, list, tuple]):
        """Set the translation component of the frame with respect to the world frame.

        Parameters
        ----------
        translation : array_like
            The translation vector with respect to the world frame.
        """
        if not isinstance(translation, Vertex):
            translation = Vertex(translation, dim=self.dim)
        if not self.parent.is_world():
            # Calculate the inverse transformation of the parent
            parent_transform_inv = self.parent.transform_to_world.inverse()
            # Set the local translation to achieve the desired global translation
            self._transform.translation = parent_transform_inv.rotation @ translation + parent_transform_inv.translation
        else:
            # Set the translation directly in world coordinates
            self._transform.translation = Vertex(translation, dim=self.dim)
        self._propagate_transformation()
    
    origin_in_world = global_translation
    
    @property
    def local_rotation(self) -> RotationMatrix:
        """Get the local rotation of the frame with respect to the parent frame."""
        return self.local_transform.rotation
    
    @local_rotation.setter
    def local_rotation(self, rotation_matrix: Union[ArrayLike, torch.Tensor, RotationMatrix, list, tuple]):
        """Set the rotation matrix of the frame's (local) parent transformation.

        Parameters
        ----------
        rotation_matrix : array_like
            The rotation matrix with respect to the parent frame.
            See also: [RotationMatrix][pymagbot.math.spatial.RotationMatrix]
        """
        if not isinstance(rotation_matrix, RotationMatrix):
            rotation_matrix = RotationMatrix(rotation_matrix, dim=self.dim)
        self._transform.rotation = rotation_matrix
        self._propagate_transformation()
    
    @property
    def global_rotation(self) -> RotationMatrix:
        """Get the rotation matrix of the frame with respect to the world frame."""
        return self.transform_to_world.rotation
    
    @global_rotation.setter
    def global_rotation(self, rotation_matrix: Union[ArrayLike, torch.Tensor, RotationMatrix, list, tuple]):
        """Set the rotation matrix of the frame with respect to the world frame.

        Parameters
        ----------
        rotation_matrix : array_like
            The rotation matrix with respect to the world frame.
        """
        if not isinstance(rotation_matrix, RotationMatrix):
            rotation_matrix = RotationMatrix(rotation_matrix, dim=self.dim)
        
        if not self.parent.is_world():
            # Calculate the inverse transformation of the parent
            parent_transform_inv = self.parent.transform_to_world.inverse()
            # Set the local rotation to achieve the desired global rotation
            self._transform.rotation = rotation_matrix @ parent_transform_inv.rotation
        
        else:
            self._transform.rotation = RotationMatrix(rotation_matrix,
                                                      dim=self.dim)  # Set the rotation directly in world coordinates
        self._propagate_transformation()
    
    #
    # def move(self, translation, local: bool = True):
    #     """Move the frame to a new translation with respect to its parent or the world frame.
    #
    #     Parameters
    #     ----------
    #     translation : array_like
    #         The translation vector.
    #     local : bool, optional
    #         Whether the translation is local to the parent frame or with respect to the world frame.
    #     """
    #     if not isinstance(translation, Vertex):
    #         translation = Vertex(translation, dim=self.dim)
    #     if local and not self.parent.is_world():
    #         self._transform.translation = self.parent._transform.rotation @ translation + self.parent._transform.translation
    #     else:
    #         self._transform.translation = Vertex(translation,
    #                                              dim=self.dim)  # Set the translation directly in world coordinates
    #     self._propagate_transformation()
    #
    # def rotate(self, rotation, local: bool = True):
    #     """Rotate the frame  with respect to its parent or the world frame.
    #
    #     Parameters
    #     ----------
    #     rotation : array_like
    #         The rotation matrix.
    #         See also: [RotationMatrix][pymagbot.math.spatial.RotationMatrix]
    #
    #     local : bool, optional
    #         Whether the rotation is local to the parent frame or with respect to the world frame.
    #     """
    #     if not isinstance(rotation, RotationMatrix):
    #         rotation = RotationMatrix(rotation, dim=self.dim)
    #     if local and not self.parent.is_world():
    #         self._transform.rotation = rotation @ self._transform.rotation
    #     else:
    #         self._transform.rotation = RotationMatrix(rotation,
    #                                                   dim=self.dim)  # Set the rotation directly in world coordinates
    #     self._propagate_transformation()
    
    def _propagate_transformation(self):
        """Propagate transformations to the _transform to all children."""
        for child in self.children:
            child._transform.translation = self._transform.rotation @ child._transform.translation + self._transform.translation
            child._transform.rotation = self._transform.rotation @ child._transform.rotation
            child._propagate_transformation()
    
    def add_child(self, child: 'Frame'):
        """Add a child frame to the frame"""
        self.children.add(child)
    
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
            `True` if the frame is a child, `False` otherwise.
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
            `True` if the frame is an ancestor, `False` otherwise.
        """
        parent: Optional['Frame'] = self.parent
        while parent:
            if parent == frame:
                return True
            parent = parent.parent
        return False
    
    def clone(self, new_name: Optional[str] = None) -> 'Frame':
        """Create a deep copy of the Frame object.

        Parameters
        ----------
        new_name : str, optional
            If provided, the cloned frame will have this new name.  If None, a new name will be generated automatically.

        Returns
        -------
        Frame
            A deep copy of the Frame object.
        """
        new_transform = RigidTransform(rotation=self.local_transform.rotation.clone(),
                                       translation=self.local_transform.translation.clone(), dim=self.dim)
        new_name = new_name.lower() if new_name else f"{self.name}_clone_{next(self._id_counter)}"
        cloned_frame = Frame(name=new_name, parent=self.parent, transform=new_transform, dim=self.dim)
        
        # Recursively clone children
        for child in self.children:
            cloned_child = child.clone()
            cloned_child.parent = cloned_frame  # Set new parent
        return cloned_frame
    
    def clone_translated(self, translation: Union[ArrayLike, torch.Tensor, Vertex, list, tuple],
                         new_name: Optional[str] = None) -> 'Frame':
        """Create a deep copy of the Frame object and translate it to a new location.

        Parameters
        ----------
        translation: array_like
            The new location where the cloned frame origin is defined.
        new_name : str, optional
            If provided, the cloned frame will have this new name.  If None, a new name will be generated automatically.

        Returns
        -------
        Frame
            A deep copy of the Frame object.
        """
        cloned_frame = self.clone(new_name)
        cloned_frame.local_translation = translation
        return cloned_frame
    
    def clone_rotated(self, rotation: Union[ArrayLike, torch.Tensor, Vertex, list, tuple],
                      new_name: Optional[str] = None) -> 'Frame':
        """Create a deep copy of the Frame object  and rotate the cloned frame.

        Parameters
        ----------
        rotation: array_like
            The new rotation matrix with which the cloned frame is rotated.
        new_name : str, optional
            If provided, the cloned frame will have this new name.  If None, a new name will be generated automatically.

        Returns
        -------
        Frame
            A deep copy of the Frame object.
        """
        cloned_frame = self.clone(new_name)
        cloned_frame.local_rotation = rotation
        return cloned_frame
    
    def remove(self):
        """Remove the frame and its children from the known frames list. """
        if self.is_world():
            raise ValueError(f"Cannot remove '{self.name}', as it is a world frame.")
        # Remove the frame and its children from the known frames list
        del known_frames[self.name]
        for child in self.children:
            child.remove()
        # Remove the frame from its parent's children list
        if self.parent:
            self.parent.children.discard(self)
    
    def inverse(self) -> 'Frame':
        """
        Create a new frame that represents the inverse transformation of this frame.

        Returns
        -------
        Frame
            A new [`Frame`][pymagbot.math.frame.Frame] instance with the inverse transformation.
        """
        return Frame(name=self.name + "_inv", parent=self.parent, transform=self.local_transform.inverse(),
                     dim=self.dim)
    
    def as_matrix(self) -> torch.Tensor:
        """Return the homogeneous transformation matrix of this frame."""
        return self._transform.as_matrix()
    
    def __str__(self):
        """Return the string representation of the frame."""
        return f"{self.as_matrix()}"
    
    def __repr__(self):
        """Return the string representation of the frame."""
        return (f"Frame(name='{self.name}', "
                f"parent='{self.parent.name if isinstance(self.parent, Frame) else None}', "
                f"_transform=\n{self._transform!r}\n),"
                f"children={[child.name for child in self.children]})")
    
    @staticmethod
    def known() -> weakref.WeakValueDictionary:
        """Get the dictionary of known frames.

        Returns
        -------
        weakref.WeakValueDictionary
            The dictionary containing all known frames.
        """
        return known_frames
    
    def to_dict(self) -> dict:
        """Converts the  [`Frame`][pymagbot.math.frame.Frame]'s representation to  a dictionary. """
        return {"name"      : self.name,
                'dim'       : self.dim,
                '_transform': self._transform.to_dict(),
                "parent"    : self.parent.name if isinstance(self.parent, Frame) else None,
                'children'  : [child.name for child in self.children]
                }
    
    @classmethod
    def from_dict(cls, dict_obj: Optional[Union[dict, str]], known: Optional[dict] = None):
        """ Create a  [`Frame`][pymagbot.math.frame.Frame] from a dictionary.

        Parameters
        ----------
        dict_obj: dict or str
            The dictionary object.
        known: dict, optional
            A dictionary of known frames (see also `get_frame`)
        Returns
        -------
        Frame
            The [`Frame`][pymagbot.math.frame.Frame] object load from the `dict_obj`.
        Raises
        ------
        TypeError
            If the `dict_obj` is not a valid [`Frame`][pymagbot.math.frame.Frame]` structure.
        """
        dict_obj = parse_dict(dict_obj)
        if known is None:
            known = cls.known()
        name = dict_obj.get('name', None)
        dim = dict_obj.get('dim', 3)
        transform = RigidTransform.from_dict(
            dict_obj['_transform']) if '_transform' in dict_obj else RigidTransform(
            dim=dim)
        parent = None if 'parent' not in dict_obj or dict_obj['parent'] is None else get_frame(dict_obj['parent'],
                                                                                               known)
        frame = cls(name=name, parent=parent, transform=transform, dim=dim)
        if 'children' in dict_obj:
            for child_name in dict_obj['children']:
                if child_name in known:
                    child = get_frame(child_name, known)
                    if isinstance(child, Frame):
                        frame.add_child(child)
                    elif isinstance(child, dict):
                        frame.add_child(cls.from_dict(child, known))
                    else:
                        raise TypeError(f"{child_name} must be either a Frame or a dict")
        return frame
    
    def plot(self, fig=None, scale=1.0, name=None):
        """Plot the frame using [Plotly](https://plotly.com/).
        
        !!! note
            Only 3D frame is currently supported.
        
        TODO: add support for 2D frame
        
        Parameters
        ----------
        fig : plotly.graph_objects.Figure, optional
            The  [Plotly](https://plotly.com/) figure to add the frame to. If `None`, a new figure will be created.
        scale : float, optional
            The scale of the coordinate frame axes. Defaults to 1.0.
        name : str, optional
            An optional name for the frame, used in labels.

        Returns
        -------
        plotly.graph_objects.Figure
            The  [Plotly](https://plotly.com/) figure object, or the modified scene if a scene was provided as input.

        Raises
        ------
        ImportError
            If  [Plotly](https://plotly.com/) is not installed.
        ValueError
            If the frame is not 3D.
        """
        try:
            import plotly.graph_objects as go
            # import numpy as np  # Import numpy within the try block
        except ImportError:
            raise ImportError("Plotly is required for plotting.")
        
        if self.dim != 3:
            raise ValueError("Plotting is only supported for 3D frames.")
        
        T = self.as_matrix().cpu().numpy()
        origin = T[:3, 3]
        R = T[:3, :3]
        
        x_axis = R[:, 0] * scale
        y_axis = R[:, 1] * scale
        z_axis = R[:, 2] * scale
        
        # Plot the axes
        origin_point = Vertex(components=torch.tensor(origin), dim=3)
        x_direction = Direction(components=torch.tensor(x_axis), dim=3)
        y_direction = Direction(components=torch.tensor(y_axis), dim=3)
        z_direction = Direction(components=torch.tensor(z_axis), dim=3)
        
        traces = []
        traces.append(self._create_axis_trace(origin_point, name if name else self.name + "_origin", 'black', 'o'))
        traces.append(self._create_axis_trace(x_direction, name if name else self.name + "_X", 'red', origin_point))
        traces.append(self._create_axis_trace(y_direction, name if name else self.name + "_Y", 'green', origin_point))
        traces.append(self._create_axis_trace(z_direction, name if name else self.name + "_Z", 'blue', origin_point))
        
        if fig is None:
            fig = go.Figure(data=traces)
            fig.update_layout(
                scene=dict(
                    xaxis=dict(title='X'),
                    yaxis=dict(title='Y'),
                    zaxis=dict(title='Z'),
                    aspectmode='data'  # Ensures proper scaling
                )
            )
            return fig
        else:
            for trace in traces:
                fig.add_trace(trace)
            return fig
    
    def _create_axis_trace(self, axis, name, color, origin=None):
        """Helper function to create a trace for an axis (Point or Direction)."""
        try:
            import plotly.graph_objects as go
        except ImportError:
            raise ImportError("Plotly is required for plotting. Install with `pip install plotly`.")
        
        if isinstance(axis, Vertex):
            x, y, z = axis.data[0].cpu().numpy(), axis.data[1].cpu().numpy(), axis.data[2].cpu().numpy()
            trace = go.Scatter3d(
                x=[x], y=[y], z=[z], mode='markers', marker=dict(color=color, symbol='circle', size=4), name=name
            )
        elif isinstance(axis, Direction):
            if origin is None:
                origin = Vertex(torch.zeros(self.dim), dim=self.dim)
            x0, y0, z0 = origin.data[0].cpu().numpy(), origin.data[1].cpu().numpy(), origin.data[2].cpu().numpy()
            x1, y1, z1 = x0 + axis.data[0].cpu().numpy(), y0 + axis.data[1].cpu().numpy(), z0 + axis.data[
                2].cpu().numpy()
            trace = go.Scatter3d(
                x=[x0, x1], y=[y0, y1], z=[z0, z1], mode='lines', line=dict(color=color), name=name
            )
        else:
            raise TypeError("Axis must be a Point or Direction object.")
        return trace
    
    def plot_tree(self, fig=None, scale=1.0, name=None):
        """Plot the frame and all its children as a tree structure.

        Parameters
        ----------
        fig : plotly.graph_objects.Figure, optional
            The  [Plotly](https://plotly.com/) figure to add the frame to. If `None`, a new figure will be created.
        scale : float, optional
            The scale of the coordinate frame axes. Defaults to 1.0.

        Returns
        -------
        plotly.graph_objects.Figure
            The  [Plotly](https://plotly.com/) figure object.
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            raise ImportError("Plotly is required for plotting.")
        
        if fig is None:
            fig = go.Figure()
            fig.update_layout(
                scene=dict(
                    xaxis=dict(title='X'),
                    yaxis=dict(title='Y'),
                    zaxis=dict(title='Z'),
                    aspectmode='data'  # Ensures proper scaling
                )
            )
        
        fig = self.plot(fig=fig, scale=scale, name=name if name else self.name)
        for i, child in enumerate(self.children):
            child.plot_tree(fig=fig, scale=scale, name=name if name else child.name)
        if fig is None:
            return fig
        else:
            return fig


def get_frame(frame: Union[Frame, str, dict], known: Optional[dict] = None) -> Frame:
    """Try to get/find/validate a frame

    The input `frame` can be a [`Frame`][pymagbot.math.frame.Frame] object, a string or a dictionary, then:

    - If `frame` is already a [`Frame`][pymagbot.math.frame.Frame] object, then it is a valid frame.
    - If `frame` is a [`str`][str], the function first check  if it is in a `known` dict and return the corresponding
      instance.
        
        !!! note
            If `frame` is not in the `known` dictionary a `RuntimeWarning` is reported.
            
    - If frame is a [`dict`][dict], then a [`Frame`][pymagbot.math.frame.Frame] object is built from it
      (see  [`Frame.from_dict()`][pymagbot.math.frame.Frame.from_dict])

    Parameters
    ----------
    frame: Frame, str or dict
        The `frame` to check or find.
    known: dict, optional
        A dictionary of known frames.
    Returns
    -------
    Frame
        A [`Frame`][pymagbot.math.frame.Frame] object
    Raises
    ------
    ValueError
        If the structure does not match a  frame object.
    TypeError
        If the input frame does not correspond to a proper frame object.
    """
    if isinstance(frame, Frame):
        return frame
    elif frame is None or isinstance(frame, str) and (len(frame) == 0 or frame in WORLD_FRAME_NAMES):
        return Frame.world
    elif isinstance(frame, dict):
        return Frame.from_dict(frame, known)
    
    if known is not None and len(known) > 0:
        if isinstance(frame, str):
            if len(frame) > 0:
                _frame = frame
            else:
                _frame = Frame.world.name
        elif isinstance(frame, dict) and 'name' in frame:
            _frame = frame['name']
        else:
            raise TypeError(f'Frame {type(frame)} must be a `Frame` object or str or a dict.')
        
        if _frame in known:
            return known[_frame]
        else:
            RuntimeWarning(f'Frame {_frame} is not in given `known` frames.')
    
    if isinstance(frame, str):
        if ':' in frame:
            try:
                dict_obj = json.loads(frame.replace("\'", "\""))
                if isinstance(dict_obj, dict):
                    return Frame.from_dict(dict_obj, known)
            except (TypeError, json.JSONDecodeError) as e:  # Catch JSONDecodeError as well
                raise ValueError(f'"{frame}" seems not a valid frame structure.  Details: {e}')
        else:
            raise ValueError(f'Cannot guess the frame with structure {frame} without known.')
    raise TypeError('Frame must be a `Frame` object or str or a dict_object.')


# Initialize the world reference frame
world = Frame(name=WORLD_FRAME_NAMES[0], dim=3)
