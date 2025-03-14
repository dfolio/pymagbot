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


import unittest

import torch

from pymagbot.math.frame import Frame
from pymagbot.math.geometry_base import Vertex, RotationMatrix

# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestFrame(unittest.TestCase):
    
    def test_frame_creation(self):
        frame = Frame(name="TestCreation")
        self.assertEqual(frame.name, "TestCreation".lower())
    
    def test_frame_hierarchy(self):
        """Test frame parent-child relationships."""
        world = Frame()
        F1 = Frame(parent=world)
        self.assertIs(F1.parent, world, "F1 should have world as its parent")
        
        F2 = Frame(parent=F1)
        self.assertIs(F2.parent, F1, "F2 should have F1 as its parent")
        
        self.assertIn(F1, world.children, "F1 should be in world's children")
        self.assertIn(F2, F1.children, "F2 should be in F1's children")
    
    def test_frame_move_to(self):
        frame = Frame(name="TestFrame")
        translation = [1, 2, 3]
        frame.move_to(translation)
        self.assertEqual(frame.transform.d.data.tolist(), translation)
    
    def test_frame_rotate(self):
        rotation_matrix = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        frame = Frame(name="TestFrame")
        frame.rotate(RotationMatrix(rotation_matrix), local=False)
        self.assertEqual(frame.transform.R.data.tolist(), rotation_matrix)
    
    def test_frame_motion(self):
        """Test moving and rotating frames."""
        world = Frame()
        F1 = Frame(parent=world)
        
        F1.move_to([1.0, 0.0, 0.0], local=False)
        self.assertTrue(torch.allclose(F1.transform.d.data, torch.tensor([1.0, 0.0, 0.0]).to(device)),
                        "F1 should be at (1,0,0)")
        
        R = RotationMatrix(torch.eye(3))
        F1.rotate(R, local=False)
        self.assertTrue(torch.allclose(F1.transform.R.data, torch.eye(3).to(device)),
                        "F1 should have identity rotation")
    
    def test_frame_inverse(self):
        frame = Frame(name="TestFrame")
        translation = [1, 2, 3]
        frame.move_to(translation)
        inverse_frame = frame.inverse()
        # self.assertEqual(inverse_frame.name, "TestFrame".lower()) # after inverse the
        self.assertEqual(inverse_frame.transform.d.data.tolist(), [-translation[0], -translation[1], -translation[2]])
    
    def test_frame_remove(self):
        frame = Frame(name="TestRemove")
        frame.remove()
        self.assertNotIn("TestRemove".lower(), Frame.known())
    
    def test_frame_children(self):
        frame = Frame(name="TestChildren")
        child = Frame(name="ChildFrame", parent=frame)
        frame.children.add(child)
        self.assertIn(child, frame.children)
    
    def test_frame_parent(self):
        frame = Frame(name="TestParent")
        parent = Frame(name="ParentFrame")
        frame.parent = parent
        self.assertEqual(frame.parent.name, "ParentFrame".lower())


class TestRotationMatrix(unittest.TestCase):
    
    def test_rotation_matrix_creation(self):
        rotation_matrix = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        rm = RotationMatrix(rotation_matrix)
        self.assertEqual(rm.data.tolist(), rotation_matrix)


class TestPoint(unittest.TestCase):
    
    def test_point_creation(self):
        point = Vertex([1, 2, 3])
        self.assertEqual(point.data.tolist(), [1, 2, 3])


if __name__ == '__main__':
    unittest.main()
