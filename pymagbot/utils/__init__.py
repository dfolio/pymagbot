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
"""utils - utility module

This package provides a collection of general utilities for PyMagBot, including:
- IO operations (serialization/deserialization, file save/load)
- Plotting utilities for visualization
- Configuration management
- Logging utilities
- Input validation functions
"""

from pymagbot.utils.config import DEVICE, DTYPE, FORMAT, USE_CUDA
from pymagbot.utils.io import (load_from_csv, load_from_json, load_from_npz, load_from_yaml, save_to_csv, save_to_json,
                               save_to_npz, save_to_yaml)

__all__ = [
    'DEVICE', 'DTYPE', 'FORMAT', 'USE_CUDA']
