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
test_frame - Unit tests for the frame submodule.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest

import torch

from pymagbot._config import DTYPE, DEVICE
from pymagbot.math.frame import Frame
from pymagbot.math.spatial import RigidTransform, RotationMatrix
from pymagbot.math.spatial_primitives import Vertex


class TestFrame(unittest.TestCase):
    
    def setUp(self):
        # Create a world frame
        self.world_frame = Frame(name="world")
        # Create a child frame with a non-trivial _transform
        self.rotation = RotationMatrix.from_euler([0.1, 0.2, 0.3])
        self.translation = Vertex([1, 2, 3])
        self.transform = RigidTransform(rotation=self.rotation, translation=self.translation)
        # Create a child frame
        self.child_frame = Frame(name="child", parent=self.world_frame, transform=self.transform)
    
    def test_frame_creation(self):
        """Test frame creation and naming."""
        self.assertEqual(Frame.world.name, "world", "World frame name should be 'world'")
        self.assertIsNotNone(self.world_frame, "World frame instance should not be None")
        self.assertIsNotNone(self.child_frame, "Child frame instance should not be None")
        self.assertEqual(self.child_frame.name, "child", "Child frame name should be 'child'")
        
        test_frame = Frame(name="TestCreation")
        self.assertEqual(test_frame.name, "testcreation", "Frame name should be lowercased")
    
    def test_frame_hierarchy(self):
        """Test frame parent-child relationships."""
        self.assertEqual(self.child_frame.parent, self.world_frame, "Child frame's parent should be world frame")
        self.assertIn(self.child_frame, self.world_frame.children, "Child frame should be in world frame's children")
        
        frame1 = Frame(parent=self.world_frame)
        self.assertIs(frame1.parent, self.world_frame, "Frame1 should have world as its parent")
        self.assertIn(frame1, self.world_frame.children, "Frame1 should be in world's children")
        
        frame2 = Frame(parent=frame1)
        self.assertIs(frame2.parent, frame1, "Frame2 should have frame1 as its parent")
        self.assertIn(frame2, frame1.children, "Frame2 should be in frame1's children")
        
    def test_frame_motion(self):
        """Test moving and rotating frames."""
        # Move the frame in global coordinates
        self.child_frame.global_translation = [1.0, 0.0, 0.0]
        torch.testing.assert_close(self.child_frame._transform.translation.data,
                                   torch.tensor([1.0, 0.0, 0.0]).to(DEVICE),
                                   atol=1e-6, rtol=1e-6,
                                   msg="Frame1 should be at (1,0,0) in world frame")
        # Rotate the frame in global coordinates
        rotation = RotationMatrix(torch.eye(3, dtype=DTYPE, device=DEVICE))
        self.child_frame.global_rotation = rotation
        torch.testing.assert_close(self.child_frame._transform.rotation.data, torch.eye(3).to(DEVICE), atol=1e-6,
                                   rtol=1e-6,
                                   msg="Frame1 should have identity rotation in world frame")
        
        new_translation = [2, 3, 4]
        self.child_frame.local_translation = new_translation
        expected_translation = Vertex(new_translation)
        torch.testing.assert_close(self.child_frame._transform.translation.data, expected_translation.data,
                                   atol=1e-6, rtol=1e-6, msg="Child frame should have moved to the new location")
        
        rotation_matrix = RotationMatrix([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        self.child_frame.global_rotation = rotation_matrix
        expected_rotation = rotation_matrix @ self.world_frame._transform.rotation
        torch.testing.assert_close(self.child_frame._transform.rotation.data, expected_rotation.data, atol=1e-6,
                                   rtol=1e-6,
                                   msg="Child frame should have rotated")
    
    def test_frame_inverse(self):
        """Test getting the inverse of the frame."""
        self.child_frame.local_translation = self.translation
        inverse_frame = self.child_frame.inverse()
        # FIXME: Inverse translation is not working correctly
        # expected_inversed_translation = [-self.translation[0], -self.translation[1], -self.translation[2]]
        # self.assertEqual(inverse_frame._transform.translation.other.tolist(), expected_inversed_translation,
        #                  "Frame correctly inversed")
        self.assertIsInstance(inverse_frame, Frame, "Inverse frame should be a Frame instance")
        self.assertEqual(inverse_frame.name, "child_inv", "Inverse frame name should be 'child_inv'")
    
    def test_frame_remove(self):
        """Test removing a frame."""
        frame = Frame(name="TestRemove")
        frame.remove()
        self.assertNotIn("testremove", Frame.known(), "Frame should be removed from known frames")
    
    def test_frame_children(self):
        """Test adding children to a frame."""
        frame = Frame(name="TestChildren")
        child = Frame(name="ChildFrame", parent=frame)
        self.assertIn(child, frame.children, "Child should be in parent's children set")
        self.assertIs(child.parent, frame, "Child's parent should be the frame")
    
    def test_frame_parent(self):
        """Test setting the parent of a frame."""
        frame = Frame(name="TestParent")
        parent = Frame(name="ParentFrame")
        frame.parent = parent
        self.assertEqual(frame.parent.name, "parentframe", "Frame's parent name should be 'ParentFrame'")
    
    def test_frame_to_dict(self):
        """Test converting the frame to a dictionary."""
        frame_dict = self.child_frame.to_dict()
        self.assertIsInstance(frame_dict, dict, "Frame dictionary should be a dictionary")
        self.assertEqual(frame_dict["name"], "child", "Frame name should match")
        self.assertEqual(frame_dict["parent"], "world", "Frame parent name should match")
        self.assertIn("_transform", frame_dict, "Frame dictionary should have a '_transform' key")
        self.assertIsInstance(frame_dict["_transform"], dict, "Transform should be a dictionary")
    
    def test_frame_from_dict(self):
        """Test creating a frame from a dictionary."""
        frame_dict = self.child_frame.to_dict()
        new_frame = Frame.from_dict(frame_dict)
        self.assertIsInstance(new_frame, Frame, "New frame should be a Frame instance")
        self.assertEqual(new_frame.name, "child", "New frame name should match original")
        self.assertEqual(new_frame.name, self.child_frame.name, "New frame name should match original")
        self.assertEqual(new_frame.parent.name, self.child_frame.parent.name, "Parent's name should match")
        # Check transformation matrices
        torch.testing.assert_close(new_frame.as_matrix(), self.child_frame.as_matrix(), atol=1e-6, rtol=1e-6,
                                   msg="Transformation matrices should match")
    
    def test_serialization_hierarchy(self):
        """Ensure parent-child relationships are maintained during serialization."""
        data = self.child_frame.to_dict()
        new_frame = Frame.from_dict(data)
        self.assertEqual(new_frame.parent.name, "world", "Parent should be correctly deserialized")
        # self.assertNotEqual(new_frame, self.child_frame, "New frame should not be the same object as original")
    
    # def test_inverse_transformation(self):
    #     """Ensure inverse transformations work correctly."""
    #     inverse_frame = self.child_frame.inverse()
    #     composed_transform = inverse_frame.as_matrix() @ self.child_frame.as_matrix()
    #     # Should be close to identity
    #     torch.testing.assert_close(composed_transform, torch.eye(4).to(DEVICE), atol=1e-6, rtol=1e-6,
    #                                msg=f"Composed _transform should be close to identity, while got {composed_transform}")


if __name__ == '__main__':
    unittest.main()
