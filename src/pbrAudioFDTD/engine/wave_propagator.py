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

from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from pbrAudioCommon.lib.import_helper import np

from ..core.entity_manager import EntityManager

from ..engine.layer_manager import LayerManager
from ..engine.fdtd_solver import FDTDManager
from ..engine.damping import Damping
from ..engine.boundary_conditions import BoundaryConditions
from ..engine.termination import SimulationTermination

@dataclass
class WavePropagator:
    """Main wave propagator manager coordinating all physical processes"""
    entity_manager: EntityManager
    idx: int

    def __post_init__(self):
        self.layer_manager = LayerManager(entity_manager=self.entity_manager, idx=self.idx)
        self.mbf_fdtd = FDTDManager(entity_manager=self.entity_manager, idx=self.idx)
        self.damping = Damping(entity_manager=self.entity_manager, idx=self.idx)
        self.boundary = BoundaryConditions(entity_manager=self.entity_manager, idx=self.idx)
        self.termination = SimulationTermination(entity_manager=self.entity_manager, idx=self.idx)

        # init layers in layer_manager
        frequencies = self.entity_manager.get('frequency_bands')
        bands = frequencies.get_bands()
        for bands_idx in range(len(bands)):
            self.layer_manager.add_new('fdtd', bands_idx)

    def update(self):
        if not self.layer_manager.ended: 
            self.mbf_fdtd.update()
#            self.damping.update()
#            self.boundary.update()
            if self.termination.update():
               self.layer_manager.ended = True
