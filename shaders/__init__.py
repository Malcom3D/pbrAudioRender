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

from .gas_shader import GasShader
from .fluid_shader import FluidShader
from .solid_shader import SolidShader

def create_shader(shader_type: str, **kwargs):
    """Factory function to create appropriate shader based on type"""
    shader_type = shader_type.lower()
    
    if shader_type == "gas":
        return GasShader(**kwargs)
    elif shader_type == "fluid":
        return FluidShader(**kwargs)
    elif shader_type == "solid":
        return SolidShader(**kwargs)
    else:
        raise ValueError(f"Unknown shader type: {shader_type}")

__all__ = [
    'GasShader',
    'FluidShader', 
    'SolidShader',
    'create_shader'
]

