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

from .spherical_source import SphericalSource
from .plane_source import PlaneSource

def create_source(source_config):
    """Factory function to create appropriate source based on configuration"""
    source_type = source_config.type.lower()
    
    if source_type == "spherical":
        return SphericalSource(source_config)
    elif source_type == "plane":
        return PlaneSource(source_config)
    else:
        raise ValueError(f"Unknown source type: {source_config.type}")

__all__ = [
    'SphericalSource',
    'PlaneSource',
    'create_source'
]

