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
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pymagbot.math.geometry_base import RotationMatrix
from pymagbot.math.frame import Frame


# from pymagbot.math.geometry import RotationMatrix, Point, Direction

def main():
    # Create a world frame
    world = Frame()
    
    # Create a child frame F1 with world as parent
    F1 = Frame(parent=world)
    
    # Move F1 in the world frame
    F1.move_to([1.0, 2.0, 3.0], local=False)
    print(f"F1 position in world frame: {F1.transform.d}")
    
    # Rotate F1 with respect to world
    rot_matrix = RotationMatrix(torch.eye(3))  # Identity rotation
    F1.rotate(rot_matrix, local=False)
    print(f"F1 rotation in world frame:\n{F1.transform.R}")
    
    # Create another frame F2 as child of F1
    F2 = Frame(parent=F1)
    
    # Move F2 locally within F1
    F2.move_to([0.5, 0.0, 0.0], local=True)
    print(f"F2 position in world frame: {F2.transform.d}")


if __name__ == "__main__":
    main()
