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
Output package for microphone and receiver simulations.
"""

from .omnidirectional_output import OmnidirectionalOutput
from .cardioid_output import CardioidOutput
from .figure8_output import Figure8Output
from .hypercardioid_output import HypercardioidOutput

# Output factory function
def create_output(output_config):
    """Factory function to create appropriate output based on configuration"""
    output_type = output_config.type.lower()
    
    if output_type == "omnidirectional":
        return OmnidirectionalOutput(output_config)
    elif output_type == "cardioid":
        return CardioidOutput(output_config)
    elif output_type == "figure8":
        return Figure8Output(output_config)
    elif output_type == "hypercardioid":
        return HypercardioidOutput(output_config)
    else:
        raise ValueError(f"Unknown output type: {output_config.type}")

__all__ = [
    'OmnidirectionalOutput',
    'CardioidOutput', 
    'Figure8Output',
    'HypercardioidOutput',
    'create_output'
]

