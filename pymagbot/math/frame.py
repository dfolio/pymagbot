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
import weakref
from typing import Optional

import torch

from pymagbot.math.geometry_base import Direction, Point, RigidTransform, RotationMatrix

__all__ = ['Frame']

# Dictionary to track known frames (ensuring uniqueness)
known_frames: weakref.WeakValueDictionary[str, 'Frame'] = weakref.WeakValueDictionary()  # Prevent memory leaks

WORLD_FRAME_NAMES: tuple[str, ...] = ('world', 'Frame_0', 'F0', 'workspace')
"""Known reserved name and aliases for world reference frame"""


class Frame():
    """
    An frame defined by a rigid transformation (SE(2) or SE(3)).
    
    This class represents a inertia frame in 2D or 3D space.  It manages the frame's
    transformation relative to its parent frame, and maintains a registry of known frames.
    """
    _id_counter = itertools.count(1)  # Unique ID generator for frames
    
    world = None
    
    def __new__(cls, name: Optional[str] = None, parent: Optional['Frame'] = None,
                transform: Optional[RigidTransform] = None, dim: int = 3) -> 'Frame':
        """
        Create a new Frame instance or retrieve an existing one with the same name.

        This method ensures that only one [`Frame`][pymagbot.math.frame.Frame] instance exists for each unique name.

        Parameters
        ----------
        name : str, optional
            The name of the frame. If None, a unique name is generated.
        parent : Frame, optional
            The parent frame. If None, the world frame is used as the parent.
        transform : RigidTransform, optional
            The rigid transformation. If None, an identity transformation is used.
        dim : int, optional
            The dimension of the frame (2 or 3). Defaults to 3.

        Returns
        -------
        Frame
            The [`Frame`][pymagbot.math.frame.Frame] instance.  If a frame with the given name already exists, that instance is returned.
        """
        # name = cls._generate_unique_name(name)
        
        name = name.lower() if name else f"Frame_{next(cls._id_counter)}"
        if name in known_frames:
            return known_frames[name]
        instance = super().__new__(cls)
        instance._name: str = name  # Store the name early
        known_frames[name] = instance
        return instance
    
    def __init__(self, name: Optional[str] = None, parent: Optional['Frame'] = None,
                 transform: Optional[RigidTransform] = None, dim: int = 3):
        """
        Initialize the [`Frame`][pymagbot.math.frame.Frame].

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
            return  # Prevent double initialization
        
        self.name: str = self._name
        self.dim: int = dim
        self.children: set['Frame'] = set()
        self.transform: RigidTransform = transform if transform else RigidTransform(dim=dim)
        
        if self.is_world():
            if Frame.world is None:
                Frame.world = self  # Set the world frame
            elif Frame.world is not self:
                raise ValueError("Only one world frame instance is allowed.")
        
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
            True if the frame's name is in the list of world frame names, False otherwise.
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
            True if the frame is a child, False otherwise.
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
            True if the frame is an ancestor, False otherwise.
        """
        parent: Optional['Frame'] = self.parent
        while parent:
            if parent == frame:
                return True
            parent = parent.parent
        return False
    
    def move_to(self, translation, local: bool = True):
        """
        Move the frame to a new location.

        Parameters
        ----------
        translation : array_like
            The translation vector.
        local : bool, optional
            Whether the translation is local to the parent frame or with respect to the world frame.
        """
        if not isinstance(translation, Point):
            translation = Point(translation, dim=self.dim)
        if local and self.parent:
            self.transform.d = Point(self.parent.transform.R.data @ translation.data + self.parent.transform.d.data,
                                     dim=self.dim)
        else:
            self.transform.d = translation  # Set the translation directly in world coordinates
        self._propagate_changes()
    
    def rotate(self, rotation_matrix, local: bool = True):
        """
        Rotate the frame.

        Parameters
        ----------
        rotation_matrix : array_like
            The rotation matrix.
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
            child.transform.d = Point(self.transform.R.data @ child.transform.d.data + self.transform.d.data,
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
        return f"name={self.name}, Frame(transform={self.transform.to_matrix().cpu().numpy()}, parent={self.parent})"
    
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
    
    def plot(self, scene=None, scale=1.0, label=True, name=None):
        """Plot the frame using Plotly.

        Parameters
        ----------
        scene : plotly.graph_objects.Scene, optional
            The Plotly scene to add the frame to. If None, a new figure and scene will be created.
        scale : float, optional
            The scale of the coordinate frame axes. Defaults to 1.0.
        label : bool, optional
            Whether to display the frame name as a label. Defaults to True.

        Returns
        -------
        plotly.graph_objects.Figure or plotly.graph_objects.Scene
            The Plotly figure object, or the modified scene if a scene was provided as input.

        Raises
        ------
        ImportError
            If Plotly is not installed.
        ValueError
            If the frame is not 3D.
        """
        try:
            import plotly.graph_objects as go
            import numpy as np  # Import numpy within the try block
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
        
        if scene is None:
            fig = go.Figure()
            scene = go.Scene(
                xaxis=dict(title='X'),
                yaxis=dict(title='Y'),
                zaxis=dict(title='Z'),
                aspectmode='data'  # Ensures proper scaling
            )
            fig.update_layout(scene=scene)  # Assign the scene to the layout
        
        # Plot the axes
        origin_point = Point(data=torch.tensor(origin), dim=3)
        x_direction = Direction(data=torch.tensor(x_axis), dim=3)
        y_direction = Direction(data=torch.tensor(y_axis), dim=3)
        z_direction = Direction(data=torch.tensor(z_axis), dim=3)
        
        origin_point.plot(scene=scene, color='black', label=name if name else self.name + "_origin", size=4)
        x_direction.plot(scene=scene, origin=origin_point, color='red', scale=1,
                         label=name if name else self.name + "_X")
        y_direction.plot(scene=scene, origin=origin_point, color='green', scale=1,
                         label=name if name else self.name + "_Y")
        z_direction.plot(scene=scene, origin=origin_point, color='blue', scale=1,
                         label=name if name else self.name + "_Z")
        
        if scene is None:
            return fig
        else:
            return scene
    
    def plot_tree(self, scene=None, scale=1.0, name=None):
        """Plot the frame and all its children as a tree structure.

        Parameters
        ----------
        scene : plotly.graph_objects.Scene, optional
            The Plotly scene to add the frame to. If None, a new figure and scene will be created.
        scale : float, optional
            The scale of the coordinate frame axes. Defaults to 1.0.

        Returns
        -------
        plotly.graph_objects.Figure
            The Plotly figure object.
        """
        
        if scene is None:
            fig = go.Figure()
            scene = go.Scene(
                xaxis=dict(title='X'),
                yaxis=dict(title='Y'),
                zaxis=dict(title='Z'),
                aspectmode='data'  # Ensures proper scaling
            )
            fig.update_layout(scene=scene)
        
        scene = self.plot(scene=scene, scale=scale, name=name if name else self.name)
        for i, child in enumerate(self.children):
            child.plot_tree(scene=scene, scale=scale, name=name if name else self.name + f"_{i + 1}")
        if scene is None:
            return fig
        else:
            return fig


# Initialize the world frame
world = Frame(name=WORLD_FRAME_NAMES[0])
