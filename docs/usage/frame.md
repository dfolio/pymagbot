# Base Geometry and Frame usage

### Creating Frames
```python
from pymagbot.math.frame import Frame

world = Frame.world # The world reference frame
robot = Frame("robot", parent=world)

print(f"Robot's parent: {robot.parent.name}")
```

### Transformations
```python
from pymagbot.math.geometry_base import RotationMatrix, Point
import plotly.graph_objects as go

rot = RotationMatrix([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
translation = Point([1, 2, 3])

robot.move_to(translation)
robot.rotate(rot)

#fig = go.Figure()
robot.plot().show()
```