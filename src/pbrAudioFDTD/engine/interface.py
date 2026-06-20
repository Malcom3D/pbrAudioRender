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

from pbrAudioCommon.lib.import_helper import np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

from ..core.entity_manager import EntityManager
from ..engine.interfaces import AbsorptionInterface, ReflectionInterface, RefractionInterface, ScatteringInterface, DiffractionInterface

@dataclass
class InterfaceManager:
    """Main interface manager handling all boundary interactions with sophisticated detection"""
    entity_manager: EntityManager
    idx: int
    bands_idx: int
    
    def __post_init__(self):
        config = self.entity_manager.get('config')
        self.voxel_size = config.acoustic_domain.voxel_size

        # Initialize individual interface handlers
        self.diffraction = DiffractionInterface(self.entity_manager, self.idx, self.bands_idx)
        self.absorption = AbsorptionInterface(self.entity_manager, self.idx, self.bands_idx)
        self.refraction = RefractionInterface(self.entity_manager, self.idx, self.bands_idx)
        self.reflection = ReflectionInterface(self.entity_manager, self.idx, self.bands_idx)
        self.scattering = ScatteringInterface(self.entity_manager, self.idx, self.bands_idx)
        
        self.interaction_threshold = config.interface.interaction_threshold
        self.min_impedance_ratio = config.interface.min_impedance_ratio
        self.max_impedance_ratio = config.interface.max_impedance_ratio
        
    def detect_boundaries(self, layer_manager, soxel_grid) -> Dict[str, Any]:
        """Sophisticated boundary detection using impedance discontinuities"""
        boundaries = {
            'impedance_discontinuities': [],
            'sound_speed_discontinuities': [],
            'material_boundaries': [],
            'edge_boundaries': []
        }
        
        # Get acoustic properties
        sound_speed = soxel_grid.get_array('sound_speed')
        density = soxel_grid.get_array('density')
        
        # Calculate acoustic impedance Z = ρc
        impedance = density * sound_speed
        
        names = []
        items = list(layer_manager.layers.items())
        for index, item in items:
            if not item.name in names and item.bands_idx == self.bands_idx:
                name = item.name
                names.append(name)

                pressure = layer_manager.get_array(name, self.bands_idx, 'pressure')  # Get pressure from layers of this band
        
                for i in range(1, impedance.shape[0]-1):
                    for j in range(1, impedance.shape[1]-1):
                        for k in range(1, impedance.shape[2]-1):
                            current_impedance = impedance[i, j, k]
                    
                            # Check neighbors for impedance discontinuities
                            for di, dj, dk in [(-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1)]:
                                ni, nj, nk = i + di, j + dj, k + dk
                        
                                if (0 <= ni < impedance.shape[0] and 
                                    0 <= nj < impedance.shape[1] and 
                                    0 <= nk < impedance.shape[2]):
                            
                                    neighbor_impedance = impedance[ni, nj, nk]
                            
                                    # Calculate impedance ratio
                                    if current_impedance > 0 and neighbor_impedance > 0:
                                        impedance_ratio = max(current_impedance, neighbor_impedance) / min(current_impedance, neighbor_impedance)
                                        
                                        # Check if this is a significant discontinuity
                                        if (impedance_ratio > self.min_impedance_ratio and 
 
                                            impedance_ratio < self.max_impedance_ratio and
                                            np.abs(pressure[i, j, k]) > self.interaction_threshold):
                                    
                                            boundaries['impedance_discontinuities'].append({
                                                'position': (i, j, k),
                                                'neighbor_position': (ni, nj, nk),
                                                'impedance_ratio': impedance_ratio,
                                                'pressure': pressure[i, j, k],
                                                'type': 'impedance'
                                            })
                            
                                    # Check sound speed discontinuities for refraction
                                    current_sound_speed = sound_speed[i, j, k]
                                    neighbor_sound_speed = sound_speed[ni, nj, nk]
                            
                                    if current_sound_speed > 0 and neighbor_sound_speed > 0:
                                        sound_speed_ratio = max(current_sound_speed, neighbor_sound_speed) / min(current_sound_speed, neighbor_sound_speed)
                                
                                        if sound_speed_ratio > 1.1:  # Significant sound speed change
                                            boundaries['sound_speed_discontinuities'].append({
                                                'position': (i, j, k),
                                                'neighbor_position': (ni, nj, nk),
                                                'sound_speed_ratio': sound_speed_ratio,
                                                'type': 'sound_speed'
                                            })
        
                # Detect edge boundaries for diffraction
                boundaries['edge_boundaries'] = self._detect_edge_boundaries(soxel_grid, pressure)
        
        return boundaries
    
    def _detect_edge_boundaries(self, soxel_grid, pressure: np.ndarray) -> List[Dict[str, Any]]:
        """Detect edge boundaries for diffraction analysis"""
        edge_boundaries = []
        
        for i in range(1, pressure.shape[0]-1):
            for j in range(1, pressure.shape[1]-1):
                for k in range(1, pressure.shape[2]-1):
                    # Check if this is an edge (object boundary with free space neighbors)
                    current_soxel = soxel_grid.soxels[i, j, k]
                    
                    if current_soxel.type == 2:  # Object
                        # Count free space neighbors
                        free_space_neighbors = 0
                        object_neighbors = 0
                        
                        for di in [-1, 0, 1]:
                            for dj in [-1, 0, 1]:
                                for dk in [-1, 0, 1]:
                                    if di == 0 and dj == 0 and dk == 0:
                                        continue
                                        
                                    ni, nj, nk = i + di, j + dj, k + dk
                                    
                                    if (0 <= ni < pressure.shape[0] and 
                                        0 <= nj < pressure.shape[1] and 
                                        0 <= nk < pressure.shape[2]):
                                        
                                        neighbor_soxel = soxel_grid.soxels[ni, nj, nk]
                                        if neighbor_soxel.type == 0:  # Free space
                                            free_space_neighbors += 1
                                        elif neighbor_soxel.type == 2:  # Object
                                            object_neighbors += 1
                        
                        # This is an edge if it has both object and free space neighbors
                        if free_space_neighbors > 0 and object_neighbors > 0:
                            # Calculate obstacle size based on connected object voxels
                            obstacle_size = self._calculate_obstacle_size(soxel_grid, (i, j, k))
                            
                            edge_boundaries.append({
                                'position': (i, j, k),
                                'obstacle_size': obstacle_size,
                                'free_space_neighbors': free_space_neighbors,
                                'object_neighbors': object_neighbors,
                                'pressure': pressure[i, j, k],
                                'type': 'edge'
                            })
        
        return edge_boundaries
    
    def _calculate_obstacle_size(self, soxel_grid, position: Tuple[int, int, int]) -> float:
        """Calculate the characteristic size of the obstacle at the given edge position"""
        i, j, k = position
        visited = set()
        object_voxels = []
        
        # Flood fill to find connected object voxels
        stack = [position]
        while stack:
            current_pos = stack.pop()
            if current_pos in visited:
                continue
                
            visited.add(current_pos)
            ci, cj, ck = current_pos
            
            # Check if this is an object voxel
            if (0 <= ci < soxel_grid.soxels.shape[0] and 
                0 <= cj < soxel_grid.soxels.shape[1] and 
                0 <= ck < soxel_grid.soxels.shape[2] and
                soxel_grid.soxels[ci, cj, ck].type == 2):
                
                object_voxels.append(current_pos)
                
                # Add neighbors to stack
                for di, dj, dk in [(-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1)]:
                    neighbor_pos = (ci + di, cj + dj, ck + dk)
                    if neighbor_pos not in visited:
                        stack.append(neighbor_pos)
        
        if not object_voxels:
            return 0.0
        
        # Calculate bounding box and characteristic size
        object_array = np.array(object_voxels)
        min_coords = np.min(object_array, axis=0)
        max_coords = np.max(object_array, axis=0)
        
        # Characteristic size as the geometric mean of dimensions
        dimensions = max_coords - min_coords + 1
        characteristic_size = np.prod(dimensions) ** (1/3)
        
        return characteristic_size * self.voxel_size  # Convert to world units

    def update(self):
        """Apply all interface interactions with proper layer management"""
        soxel_grid = self.entity_manager.get('soxel_grid')
        wave_propagator = self.entity_manager.get('wave_propagators', self.idx)
        layer_manager = wave_propagator.layer_manager

        boundaries = self.detect_boundaries(layer_manager, soxel_grid)

        # Apply interfaces if we have significant boundaries and sufficient pressure
        pressure = layer_manager.get_array('fdtd', self.bands_idx, 'pressure')
        max_pressure = np.max(np.abs(pressure))

        if not (len(boundaries['impedance_discontinuities']) > 0 or 
                len(boundaries['sound_speed_discontinuities']) > 0 or
                len(boundaries['edge_boundaries']) > 0) and max_pressure > self.interaction_threshold:
            return
        
        # Apply interface interactions in physically correct order
        
        # 11. Diffraction (occurs at edges before other interactions)
        if boundaries['edge_boundaries']:
            self.diffraction.update(boundaries)

        # 2. Absorption (energy loss at boundaries)
        if boundaries['impedance_discontinuities']:
            self.absorption.update(boundaries)

        # 3. Refraction (wave propagation through boundaries)
        if boundaries['sound_speed_discontinuities']:
            self.refraction.update(boundaries)

        # 4. Reflection with layer management
        if boundaries['impedance_discontinuities']:
            self.reflection.update(boundaries)

        # 5. Scattering over reflection layers
        if boundaries['impedance_discontinuities']:
            self.scattering.update(boundaries)
