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

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

from ..core.entity_manager import EntityManager
from ..engine.interfaces import AbsorptionInterface, ReflectionInterface, RefractionInterface, ScatteringInterface, DiffractionInterface

@dataclass
class InterfaceManager:
    """Main interface manager handling all boundary interactions with sophisticated detection"""
    entity_manager: EntityManager

    def __post_init__(self):
        config = self.entity_manager.get('config')

        # Initialize individual interface handlers
        self.diffraction = DiffractionInterface(self.entity_manager)
        self.absorption = AbsorptionInterface(self.entity_manager)
        self.refraction = RefractionInterface(self.entity_manager)
        self.reflection = ReflectionInterface(self.entity_manager)
        self.scattering = ScatteringInterface(self.entity_manager)

        self.interaction_threshold = config.interface.interaction_threshold
        self.min_impedance_ratio = config.interface.min_impedance_ratio
        self.max_impedance_ratio = config.interface.max_impedance_ratio

    def compute(self):
        pass
