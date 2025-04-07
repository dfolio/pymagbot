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
test_spatial - Unit tests for the spatial submodule.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest

import torch

from pymagbot.utils.config import DTYPE, TENSOR_PRINT_FORMAT, DEVICE
from pymagbot.math.spatial import RigidTransform, RotationMatrix
from pymagbot.math.spatial.primitives import Vertex, Direction


class TestRotationMatrix(unittest.TestCase):
    def setUp(self):
        """Set up test cases."""
        self.eye3 = torch.eye(3, device=DEVICE, dtype=DTYPE)
        self.identity = RotationMatrix()
        self.rotation_matrix = torch.tensor([[0.866, -0.5, 0],
                                             [0.5, 0.866, 0],
                                             [0, 0, 1]], dtype=DTYPE, device=DEVICE)
        self.euler_rotation = RotationMatrix.from_euler([0.1, 0.2, 0.3])
        # self.axis_rotation = RotationMatrix.from_axis_angle([1, 0, 0], 0.5)
        self.small_angle = 1e-8
        self.singular_matrix = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], device=DEVICE, dtype=DTYPE)
    
    def test_identity(self):
        """Test that the default RotationMatrix is the identity matrix."""
        torch.testing.assert_close(self.identity._data, self.eye3, atol=1e-6, rtol=1e-6)
    
    def test_rotation_matrix_creation(self):
        rotation_matrix = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        rm = RotationMatrix(rotation_matrix)
        self.assertEqual(rm.data.tolist(), rotation_matrix)

        rotation = RotationMatrix(self.rotation_matrix)
        self.assertTrue(torch.allclose(rotation.data, self.rotation_matrix))
        self.assertEqual(rotation.dim, 3)
    
    # def test_from_euler(self):
    #     """Test Euler angle conversion."""
    #     r1 = RotationMatrix.from_euler([0.1, 0.2, 0.3])
    #     r2 = RotationMatrix.from_quaternion(r1.to_quaternion())
    #     torch.testing.assert_close(r1._data, r2._data, atol=1e-6, rtol=1e-6)
    
    # def test_from_axis_angle(self):
    #     """Test axis-angle rotation."""
    #     axis = torch.tensor([1.0, 0.0, 0.0])
    #     angle = torch.tensor(0.5)
    #     _rotation = RotationMatrix.from_axis_angle(axis, angle)
    #     self.assertAlmostEqual(torch.det(_rotation._data).item(), 1.0, places=6)  # Ensure valid rotation
    
    def test_inverse(self):
        """Test matrix inversion."""
        inv_r = self.euler_rotation.inverse()
        identity = self.euler_rotation @ inv_r
        torch.testing.assert_close(identity.data, self.eye3, atol=1e-6, rtol=1e-6)
        
        r1 = RotationMatrix.from_quaternion([0.1, 0.2, 0.3, 1])
        r2 = RotationMatrix.from_quaternion([-0.1, -0.2, -0.3, 1])
        r_combined = r1 @ r2
        self.assertTrue(torch.allclose(r_combined.data, self.identity.data, atol=1e-6, rtol=1e-6))
    
    def test_is_orthogonal(self):
        """Ensure the matrix is orthogonal."""
        self.assertTrue(self.euler_rotation.is_orthogonal())
    
    def test_serialization(self):
        """Test serialization and deserialization of RotationMatrix."""
        data = self.euler_rotation.to_dict()
        restored = RotationMatrix.from_dict(data)
        torch.testing.assert_close(restored._data, self.euler_rotation._data, atol=1e-6, rtol=1e-6)
    
    def test_str_representation(self):
        """Test string representation with global formatting."""
        TENSOR_PRINT_FORMAT["precision"] = 2
        r_str = repr(self.euler_rotation)
        self.assertIn("RotationMatrix(", r_str)  # Ensure class name appears
    
    def test_singular_matrix_handling(self):
        """Ensure singular matrices are correctly identified and raise expected errors."""
        with self.assertRaises(ValueError):
            RotationMatrix(self.singular_matrix)
    
    def test_small_angle_approximation(self):
        """Test behavior for very small rotations."""
        r = RotationMatrix.from_euler([self.small_angle, 0, 0])
        expected = torch.tensor([[1, 0, 0], [0, 1, -self.small_angle], [0, self.small_angle, 1]],
                                dtype=DTYPE, device=DEVICE)
        self.assertTrue(torch.allclose(r.data, expected, atol=1e-8))
    
    def test_quaternion(self):
        """Verify conversion from rotation matrix to quaternion."""
        
        r = RotationMatrix.from_euler([0.1, 0.2, 0.3])
        q = r.to_quaternion()
        self.assertEqual(q.shape, (4,))  # Ensure quaternion has correct shape
        self.assertAlmostEqual(torch.norm(q).item(), 1.0, places=6)  # Should be unit norm
    
    def test_transformation(self):
        v = Vertex([1.0, 0.0, 0.0])
        r = RotationMatrix.from_euler([0, 0, torch.pi / 2])
        v_rotated = r @ v
        expected = torch.tensor([0.0, 1.0, 0.0], dtype=DTYPE, device=DEVICE)
        self.assertTrue(torch.allclose(v_rotated.data, expected, atol=1e-6, rtol=1e-6))


class TestRigidTransform(unittest.TestCase):
    
    def setUp(self):
        """Create a rigid transformation"""
        self.eye3 = torch.eye(3, device=DEVICE, dtype=DTYPE)
        self.eye4 = torch.eye(4, device=DEVICE, dtype=DTYPE)
        self.identity = RigidTransform()
        self.R = RotationMatrix.from_euler([0.1, -0.2, 0.3])
        self.d = Vertex([1.0, 2.0, 3.0])
        self.rigid_transform = RigidTransform(rotation=self.R, translation=self.d)
    
    def test_rigid_transform_initialization(self):
        """Test if the rigid transformation is initialized correctly"""
        self.assertIsNotNone(self.rigid_transform)
        self.assertIsInstance(self.rigid_transform.rotation, RotationMatrix)
        self.assertIsInstance(self.rigid_transform.translation, Vertex)
        
        matrix = self.rigid_transform.as_matrix()
        self.assertTrue(torch.allclose(matrix[:3, :3], self.R.data, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(matrix[:3, 3], self.d.data, atol=1e-6, rtol=1e-6))
    
    def test_identity_transform(self):
        """Test that the default RigidTransform is identity."""
        torch.testing.assert_close(self.identity.as_matrix(), self.eye4, atol=1e-6, rtol=1e-6)
    
    def test_rigid_transform_inverse(self):
        """Test getting the inverse of the rigid transformation"""
        inverse_transform = self.rigid_transform.inverse()
        self.assertIsInstance(inverse_transform, RigidTransform)
        identity = inverse_transform.as_matrix() @ self.rigid_transform.as_matrix()
        self.assertTrue(torch.allclose(identity, self.eye4, atol=1e-6, rtol=1e-6))
    
    def test_translation_preserved(self):
        """Check if translation is stored correctly."""
        torch.testing.assert_close(self.rigid_transform.translation.data, self.d.data)
    
    def test_rotation_preserved(self):
        """Check if rotation is stored correctly."""
        torch.testing.assert_close(self.rigid_transform.rotation.data, self.R.data)
    
    def test_serialization(self):
        """Ensure serialization/deserialization works for RigidTransform."""
        data = self.rigid_transform.to_dict()
        restored = RigidTransform.from_dict(data)
        torch.testing.assert_close(restored.as_matrix(), self.rigid_transform.as_matrix(), atol=1e-6, rtol=1e-6)
    
    def test_transform_point(self):
        """Apply transformation to a point."""
        point = Vertex([1.0, 0.0, 0.0])
        transformed_point = self.rigid_transform.apply_to(point)
        expected_point = self.rigid_transform.as_matrix() @ point.homogeneous()
        torch.testing.assert_close(transformed_point.data, expected_point[:3])
    
    def test_composition(self):
        """Test composition of two transformations."""
        T1 = RigidTransform(rotation=RotationMatrix.from_euler([0.2, 0.1, -0.3]), translation=[2.0, -1.0, 0.5])
        T2 = RigidTransform(rotation=RotationMatrix.from_euler([-0.1, 0.3, 0.2]), translation=[-1.0, 2.5, -0.5])
        composed = RigidTransform.from_matrix(T1.as_matrix() @ T2.as_matrix())
        expected = T1.as_matrix() @ T2.as_matrix()
        torch.testing.assert_close(composed.as_matrix(), expected, atol=1e-6, rtol=1e-6)


class TestSpatialPrimitives(unittest.TestCase):
    """Unit tests for Vertex and Direction classes."""
    
    def setUp(self):
        """Initialize common test variables."""
        self.point_data = torch.tensor([1.0, 2.0, 3.0], dtype=DTYPE, device=DEVICE)
        self.vertex = Vertex([1.0, 2.0, 3.0])
        self.direction = Direction([0.0, 1.0, 0.0])
    
    def test_vertex_init(self):
        """Test initialization of Vertex."""
        point = Vertex(self.point_data)
        self.assertTrue(torch.allclose(point.data, self.point_data))
        self.assertEqual(point.dim, 3)
        
    def test_vertex_translation(self):
        """Test translation of a vertex."""
        translation = torch.tensor([1.0, -1.0, 2.0], dtype=DTYPE, device=DEVICE)
        new_vertex: Vertex = self.vertex + translation
        expected = torch.tensor([2.0, 1.0, 5.0], dtype=DTYPE, device=DEVICE)
        torch.testing.assert_close(new_vertex.components.data, expected)

    def test_vertex_transform(self):
        """Test transforming a Vertex using RigidTransform."""
        point = Vertex(self.point_data)
        transform = RigidTransform(rotation=RotationMatrix.from_euler([0.2, 0.1, -0.3]),
                                   translation=[0,0,0])
        transformed_point = point.apply_rigid_transform(transform.rotation.data, transform.translation.data)
        self.assertIsInstance(transformed_point, Vertex)
    
    def test_direction_init(self):
        """Test initialization of Direction."""
        vector = Direction(self.point_data)
        self.assertTrue(torch.allclose(vector.data, self.point_data / torch.linalg.norm(self.point_data)))
        self.assertEqual(vector.dim, 3)
        
    def test_direction_normalization(self):
        """Test that direction components are properly normalized."""
        expected = torch.tensor([0.0, 1.0, 0.0], dtype=DTYPE, device=DEVICE)
        torch.testing.assert_close(self.direction.components, expected)
        self.direction.components = [0.0, 10.0, 0.0]
        torch.testing.assert_close(self.direction.components, expected)


if __name__ == '__main__':
    unittest.main()
