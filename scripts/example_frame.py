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

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pymagbot.math.spatial import RotationMatrix
from pymagbot.math.spatial.frame import Frame


# from pymagbot.math.geometry import RotationMatrix, Point, Direction

def main():
    # Create a world frame
    world = Frame.world
    
    # Create a child frame F1 with world as parent
    F1 = Frame(parent=world)
    
    # Move F1 in the world frame
    F1.global_translation = [1.0, 2.0, 3.0]
    print(f"{F1.name} position in world frame: {F1._transform.translation}")
    
    # Rotate F1 with respect to world
    rot_matrix = RotationMatrix([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    F1.global_rotation = rot_matrix
    print(f"{F1.name} rotation in world frame:\n{F1._transform.rotation}")
    
    # Create another frame F2 as child of F1
    F2 = Frame(parent=F1)
    
    # Move F2 locally within F1
    F2.local_translation = [0.5, 0.0, 0.0]
    print(f"{F2.name} position in world frame: {F2._transform.translation}")
    
    world.plot_tree().show()


if __name__ == "__main__":
    main()
