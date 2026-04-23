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

from dataclasses import dataclass, field
from ..core.entity_manager import EntityManager

@dataclass
class FrameCounter:
    """Manage Current Sound Frame"""
    entity_manager: EntityManager

    def __post_init__(self):
        self.current_frame = 0
        config = self.entity_manager.get('config')
        if hasattr(config.system, 'frame_limit'):
            self.frame_limit = config.system.frame_limit

    def next(self) -> int:
        next_frame = self.current_frame + 1
        if self.frame_limit >= next_frame:
            self.current_frame = next_frame
        return self.current_frame

    def get(self) -> int:
        return self.current_frame

    def get_limit(self) -> int:
        return self.frame_limit
