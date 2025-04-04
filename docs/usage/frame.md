## Frame usage

TODO: further example of frames usage

### Creating Frames

```python
from pymagbot.math.spatial.frame import Frame

world = Frame.world  # The world reference frame
robot = Frame("robot", parent=world)

print(f"Robot's parent: {robot.parent.name}")
```
