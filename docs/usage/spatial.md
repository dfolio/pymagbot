## Spatial transformation and primitive usage

### Creating and Using Rotation Matrices

```python
from pymagbot.math.spatial import RotationMatrix

# Create a rotation matrix from Euler angles
rot_mat = RotationMatrix.from_euler([0.5, 0.3, 0.1])
print("Rotation Matrix:", rot_mat)

# Convert rotation matrix to quaternion
q = rot_mat.to_quaternion()
print("Quaternion:", q)
```

### Applying Rigid Transformations
```python
from pymagbot.math.spatial import RigidTransform, RotationMatrix

# Define rotation and translation
rot_mat = RotationMatrix.from_euler([0.1, 0.2, 0.3])
d = [1.0, 2.0, 3.0]

# Create a rigid _transform
T = RigidTransform(rotation=rot_mat, translation=d)
print("Rigid Transform Matrix:", T.as_matrix())
```

### Working with Spatial Primitives
```python
from pymagbot.math.spatial_primitives import Vertex, Direction

# Define a vertex (point in space)
p = Vertex([1, 2, 3])
print("Vertex:", p)

# Define a direction vector (unit vector)
d = Direction([1, 1, 0])
print("Direction:", d)
```

### Transformations

```python
from pymagbot.math.spatial import RotationMatrix, Vertex
import plotly.graph_objects as go

rot = RotationMatrix([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
translation = Vertex([1, 2, 3])
```