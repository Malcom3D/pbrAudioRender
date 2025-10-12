# Copyright (C) 2025 Malcom3D <malcom3d.gpl@gmail.com>
#
# This file is part of pbrAudio.
#
# pbrAudio is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# pbrAudio is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with pbrAudio.  If not, see <https://www.gnu.org/licenses/>.
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Utility classes and configuration
"""

from .config import Config, SystemConfig, VoxelGridConfig, SourceConfig, OutputConfig, ObjectConfig
from .gpu_acceleration import GPUManager, GPUConfig, GPUArrayManager
from .parallel_proc import ParallelProcessor, setup_array_backend, configure_numba

__all__ = [
    'Config', 'SystemConfig', 'VoxelGridConfig', 'SourceConfig', 'OutputConfig', 'ObjectConfig',
    'GPUManager', 'GPUConfig', 'GPUArrayManager', 
    'ParallelProcessor', 'setup_array_backend', 'configure_numba'
]

