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
"""Define default behaviour and global variables in `pymagbot`"""


import torch


# Check if CUDA is available
if torch.cuda.is_available():
    USE_CUDA = True
    """ Global flag for (cuda) GPU availability """
    DEVICE = torch.device('cuda')
    """ Global torch device (cuda or cpu) """
else:
    USE_CUDA = False
    DEVICE = torch.device('cpu')

torch.set_default_device(DEVICE)

USE_FLOAT64 = False  # Set to True for double precision
DTYPE = torch.float64 if USE_FLOAT64 else torch.float32
"""Global type used for precision (float32 or float64).

Note:
-----
Operations on [`torch.float64`][torch.float64] tensors are generally slower than on  [`torch.float32`][torch.float32]
tensors on GPUs.
"""
torch.set_default_dtype(DTYPE)

FORMAT: dict = {
    "precision"    : 3,  # Number of decimal places
    "sci_mode"     : False,  # Use scientific notation
    "align_columns": True,  # Align columns for better readability
    "max_width"    : 80  # Maximum width before line wrapping
}
""" Default format for printing tensors """

__author__ = "D. Folio"
