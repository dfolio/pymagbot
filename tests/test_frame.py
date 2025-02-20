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

import unittest

import torch

from pymagbot.math.frame import Frame, RotationMatrix, Point

# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestFrame(unittest.TestCase):
    
    def test_frame_creation(self):
        frame = Frame(name="TestFrame")
        self.assertEqual(frame.name, "TestFrame")
    
    def test_frame_hierarchy(self):
        """Test frame parent-child relationships."""
        world = Frame()
        F1 = Frame(parent=world)
        self.assertIs(F1.parent, world, "F1 should have world as its parent")
        
        F2 = Frame(parent=F1)
        self.assertIs(F2.parent, F1, "F2 should have F1 as its parent")
        
        self.assertIn(F1, world.children, "F1 should be in world's children")
        self.assertIn(F2, F1.children, "F2 should be in F1's children")
    
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


class TestRotationMatrix(unittest.TestCase):

    def test_rotation_matrix_creation(self):
        rotation_matrix = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        rm = RotationMatrix(rotation_matrix)
        self.assertEqual(rm.data.tolist(), rotation_matrix)

class TestPoint(unittest.TestCase):

    def test_point_creation(self):
        point = Point([1, 2, 3])
        self.assertEqual(point.data.tolist(), [1, 2, 3])
if __name__ == '__main__':
    unittest.main()
