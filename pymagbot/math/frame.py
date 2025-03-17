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

# import numpy as np
# from scipy.spatial.transform import Rotation as R
import itertools
import weakref
from typing import Optional

import torch

from pymagbot.math.geometry_base import Direction, RigidTransform, RotationMatrix, Vertex

__all__ = ['Frame']

known_frames: weakref.WeakValueDictionary[str, 'Frame'] = weakref.WeakValueDictionary()  # Prevent memory leaks
"""Dictionary to track known frames and ensuring uniqueness."""

WORLD_FRAME_NAMES: tuple[str, ...] = ('world', 'Frame_0', 'F0', 'workspace')
"""Known reserved name and aliases for world frame of reference"""


class Frame:
    """
    A frame defined by a [rigid transformation](https://en.wikipedia.org/wiki/Rigid_transformation) (SE(2) or SE(3)).
    
    This class is used to represent and handle **frames of reference** (or reference frames) either in 2D or 3D space.
    In physics, a frame is an abstract coordinate system. This means it has a starting point
    (i.e. [`Vertex`][pymagbot.math.geometry_base.Vertex]) and [`Direction`][pymagbot.math.geometry_base.Direction]
    specified in physical space. Specifically, the [`Frame`][pymagbot.math.frame.Frame] class manages _moving frames_.
    This is a frame of reference that moves with the observer (i.e. the  [`parent`][pymagbot.math.frame.Frame(parent)])
    along a trajectory (i.e. a curve).
    
    Also, each [`Frame`][pymagbot.math.frame.Frame] has a **unique** (case-insensitive) name.
    This name is stored in a dictionary called [`known_frames`][pymagbot.math.frame.known_frames].
    There is a sole global [world frame][pymagbot.math.frame.Frame.world] which is the parent (or ancestor) of all the
    others.
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
        name = name.lower() if name else f"Frame_{next(cls._id_counter)}"
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
            The name of the frame. The name of a frame must be unique  (case-insensitive).
            If `None`, a name is automatically generated.
        parent : Frame, optional
            The parent frame. If `None`, the [world][pymagbot.math.frame.Frame.world] frame is used as the parent.
        transform : RigidTransform, optional
            The rigid transformation of the frame with respect to the parent frame.
            If `None`, an identity transformation is used.
            
            !!! seealso
                [`RigidTransform`][pymagbot.math.geometry_base.RigidTransform] class for details about rigid
                transformation.
            
        dim : int, optional
            The dimension of the frame (e.g. 2 or 3).
        
        !!! seealso
            [`RigidTransform`][pymagbot.math.geometry_base.RigidTransform] class for details about rigid transformation.
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
                raise ValueError(f"Only one world frame instance is allowed: {self.name:}.")
        
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
            `True` if the frame's name is in the list of world frame names, `False` otherwise.
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
    
    def move_to(self, location, local: bool = True):
        """
        Move the frame to a new location.

        Parameters
        ----------
        location : array_like
            The translation vector.
        local : bool, optional
            Whether the translation is local to the parent frame or with respect to the world frame.
        """
        if not isinstance(location, Vertex):
            location = Vertex(location, dim=self.dim)
        if local and self.parent:
            self.transform.d = Vertex(self.parent.transform.R.data @ location.data + self.parent.transform.d.data,
                                      dim=self.dim)
        else:
            self.transform.d = location  # Set the translation directly in world coordinates
        self._propagate_changes()
    
    def rotate(self, rotation_matrix, local: bool = True):
        """
        Rotate the frame.

        Parameters
        ----------
        rotation_matrix : array_like
            The rotation matrix.
            See also: [RotationMatrix][pymagbot.math.geometry_base.RotationMatrix]
            
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
            child.transform.d = Vertex(self.transform.R.data @ child.transform.d.data + self.transform.d.data,
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
            A new [`Frame`][pymagbot.math.frame.Frame] instance with the inverse transformation.
        """
        return Frame(name=self.name + "_inv", parent=self.parent, transform=self.transform.inverse(), dim=self.dim)
    
    def to_matrix(self):
        return self.transform.to_matrix()
    
    def __repr__(self):
        return (f"name={self.name}, parent={self.parent.name}), transform={self.transform.to_matrix().cpu().numpy()}")
    
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
        
        T = self.to_matrix().cpu().numpy()
        origin = T[:3, 3]
        R = T[:3, :3]
        
        x_axis = R[:, 0] * scale
        y_axis = R[:, 1] * scale
        z_axis = R[:, 2] * scale
        
        # Plot the axes
        origin_point = Vertex(data=torch.tensor(origin), dim=3)
        x_direction = Direction(coordinates=torch.tensor(x_axis), dim=3)
        y_direction = Direction(coordinates=torch.tensor(y_axis), dim=3)
        z_direction = Direction(coordinates=torch.tensor(z_axis), dim=3)
        
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


# Initialize the world reference frame
world = Frame(name=WORLD_FRAME_NAMES[0])
