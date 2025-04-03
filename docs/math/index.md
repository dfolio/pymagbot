# Math submodule

 Submodules for mathematical operations, spatial transformations and primitives, frames and geometric primitives.

::: pymagbot.math
    options:
      members: false


## Quick Example
```python
from pymagbot.math.spatial import RigidTransform, RotationMatrix
from pymagbot.math.spatial_primitives import Vertex, Direction

# Create a rotation matrix
R = RotationMatrix.from_euler([0.1, 0.2, 0.3])
print("Rotation Matrix:", R)

# Create a rigid transformation
T = RigidTransform(rotation=R, translation=[1.0, 2.0, 3.0])
print("Rigid Transform Matrix:", T.as_matrix())

# Define spatial primitives
p = Vertex([1, 2, 3])
d = Direction([1, 0, 0])
print("Vertex:", p)
print("Direction:", d)
```