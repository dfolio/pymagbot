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
test_geometry - Unit tests for the geometry submodule.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest

import torch
from pymagbot.utils.config import DEVICE, DTYPE
from pymagbot.math.geometry import Coordinate, Span, Mesh, EuclideanVector
from pymagbot.math.spatial.frame import Frame


class TestCoordinate(unittest.TestCase):
    def setUp(self):
        self.frame = Frame()
        self.components = torch.tensor([1.0, 2.0, 3.0], dtype=DTYPE, device=DEVICE)
        self.coord = Coordinate(components=self.components, frame=self.frame)
    
    def test_initialization(self):
        self.assertEqual(self.coord.frame, self.frame)
        torch.testing.assert_close(self.coord.components.data, self.components)
    
    def test_express(self):
        new_frame = self.frame.clone_translated([1, 1, 1])
        transformed_coord = self.coord.express_in_frame(new_frame)
        expected = torch.tensor([2.0, 3.0, 4.0])
        torch.testing.assert_close(transformed_coord.data, expected)


class TestSpan(unittest.TestCase):
    def setUp(self):
        self.start = 0
        self.end = 1
        self.span = Span(self.start, self.end)
    
    def test_length(self):
        torch.testing.assert_close(self.span.range, 1)


class TestMesh(unittest.TestCase):
    def setUp(self):
        self.start = 0
        self.end = 1
        self.num = 10
        self.span_x = Span(self.start, self.end, self.num)
        self.span_y = Span(self.start, self.end, self.num)
        self.span_z = Span(self.start, self.end, self.num)
        self.spans = (self.span_x, self.span_y, self.span_z)
        self.mesh3d = Mesh(self.spans, frame=Frame.world)
    
    def test_dimensions(self):
        self.assertEqual(self.mesh3d.grid.shape, (10, 10, 10, 3))


class TestEuclideanVector(unittest.TestCase):
    def setUp(self):
        self.components = torch.tensor([1.0, 2.0, 3.0], dtype=DTYPE, device=DEVICE)
        self.vector = EuclideanVector([1.0, 2.0, 3.0], frame=Frame.world)
    
    def test_norm(self):
        torch.testing.assert_close(self.vector.magnitude, torch.norm(self.components))


# class TestVectorField(unittest.TestCase):
#     def setUp(self):
#
#         self.start = 0
#         self.end = 1
#         self.num = 10
#         self.span_x = Span(self.start, self.end, self.num)
#         self.span_y = Span(self.start, self.end, self.num)
#         self.span_z = Span(self.start, self.end, self.num)
#         self.spans = (self.span_x, self.span_y, self.span_z)
#         self.mesh3d = Mesh(self.spans, frame=Frame.world)
#
#         components = torch.randn(*self.mesh3d.data.shape)
#         self.field = VectorField(coords, components, frame=Frame.world)
#
#     def test_shape(self):
#         self.assertEqual(self.field.coordinates.shape, self.field.components.shape)


if __name__ == '__main__':
    unittest.main()
