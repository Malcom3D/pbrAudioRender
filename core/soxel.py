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
import zarr
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import trimesh
import soundfile as sf

@dataclass
class Soxel:
    """Represents acoustic properties of a single voxel"""
    idx: int
    is_source: bool = False
    is_medium: bool = True
    sound_speed: float = 343.0
    density: float = 1.2
    absorption_coeffs: Dict[float, float] = None
    reflection_coeffs: Dict[float, float] = None
    scattering_coeffs: Dict[float, float] = None
    audio_sample: float = 0.0
    
    def __post_init__(self):
        if self.absorption_coeffs is None:
            self.absorption_coeffs = {1000: 0.1}
        if self.reflection_coeffs is None:
            self.reflection_coeffs = {1000: 0.9}
        if self.scattering_coeffs is None:
            self.scattering_coeffs = {1000: 0.05}
    
    def get_property_at_frequency(self, property_dict: Dict[float, float], frequency: float) -> float:
        """Get frequency-dependent property value with interpolation"""
        if not property_dict or len(property_dict) == 0:
            return 0.0
        
        freqs = np.array(list(property_dict.keys()))
        values = np.array(list(property_dict.values()))
        
        if len(freqs) == 1:
            return values[0]
        
        if frequency <= freqs[0]:
            return values[0]
        if frequency >= freqs[-1]:
            return values[-1]
        
        return float(np.interp(frequency, freqs, values))

class SoxelGrid:
    """Manages the 33D grid of Soxels"""
    
    def __init__(self, voxel_config, sources: List, objects: List):
        self.config = voxel_config
        self.sources = sources
        self.objects = objects
        self.shape = voxel_config.shape
        self.voxel_size = voxel_config.voxel_size
        self.current_time = 0.0
        
        # Initialize Soxel grid
        self.grid = self._initialize_grid()
        self.source_audio = self._load_source_audio()
        self.source_positions = self._load_source_positions()
        self.object_meshes = self._load_object_meshes()
        
    def _initialize_grid(self) -> np.ndarray:
        """Initialize the 3D Soxel grid with default medium"""
        grid = np.empty(self.shape, dtype=object)
        
        default_soxel = Soxel(
            idx=0,
            sound_speed=self.config.default_sound_speed,
            density=self.config.default_density
        )
        
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                for k in range(self.shape[2]):
                    grid[i, j, k] = default_soxel
        
        return grid
    
    def _load_source_audio(self) -> Dict[int, np.ndarray]:
        """Load audio data for all sources"""
        audio_data = {}
        for source in self.sources:
            try:
                # Load WAV file using soundfile
                audio, sample_rate = sf.read(source.audio_file, dtype='float32')
                if len(audio.shape) > 1:
                    audio = audio[:, 0]  # Take first first channel for mono
                audio_data[source.idx] = audio
            except Exception as e:
                print(f"Error loading audio for source {source.idx}: {e}")
                # Create silent audio as fallback
                audio_data[source.idx] = np.zeros(44100, dtype=np.float32)
        return audio_data
    
    def _load_source_positions(self) -> Dict[int, np.ndarray]:
        """Load position and rotation data for all sources"""
        positions = {}
        for source in self.sources:
            try:
                data = np.load(source.position_file)
                positions[source.idx] = data
            except Exception as e:
                print(f"Error loading positions for source {source.idx}: {e}")
                # Create default positions at center
                n_frames = len(self.source_audio.get(source.idx, np.zeros(44100)))
                positions[source.idx] = np.zeros((n_frames, 7))  # x,y,z + quaternion
                # Set default position to center of grid
                center = np.array(self.shape) * self.voxel_size / 2
                positions[source.idx][:, :3] = center
                positions[source.idx][:, 6] = 1.0  # w component of quaternion
        return positions
    
    def _load_object_meshes(self) -> Dict[str, List[trimesh.Trimesh]]:
        """Load object meshes for all frames"""
        object_meshes = {}
        for obj in self.objects:
            try:
                meshes = []
                for obj_file in obj.obj_files:
                    mesh = trimesh.load_mesh(obj_file)
                    meshes.append(mesh)
                object_meshes[obj.name] = meshes
            except Exception as e:
                print(f"Error loading mesh for object {obj.name}: {e}")
                object_meshes[obj.name] = []
        return object_meshes
    
    def update(self, current_frame: int):
        """Update the Soxel grid for current sound frame"""
        self.current_time = current_frame / self.config.sample_rate
        
        # Reset grid to default medium
        self._reset_to_default()
        
        # Update objects
        for obj in self.objects:
            self._update_object_voxels(obj, current_frame)
        
        # Update sources
        for source in self.sources:
            self._update_source_voxels(source, current_frame)
    
    def _reset_to_default(self):
        """Reset grid to default acoustic medium"""
        default_soxel = Soxel(
            idx=0,
            sound_speed=self.config.default_sound_speed,
            density=self.config.default_density

        )
        
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                for k in range(self.shape[2]):
                    self.grid[i, j, k] = default_soxel
    
    def _update_object_voxels(self, obj: Any, current_frame: int):
        """Update voxels occupied by objects"""
        try:
            if obj.name in self.object_meshes:
                meshes = self.object_meshes[obj.name]
                if current_frame < len(meshes):
                    mesh = meshes[current_frame]
                    voxelized = self._voxelize_mesh(mesh)
                    
                    for (i, j, k), occupied in np.ndenumerate(voxelized):
                        if occupied and self._is_in_bounds((i, j, k)):
                            self.grid[i, j, k] = Soxel(
                                idx=hash(obj.name) & 0xFFFF,  # Limit to 16-bit
                                sound_speed=obj.acoustic_shader.get('sound_speed', 5000.0),
                                density=obj.acoustic_shader.get('density', 2000.0),
                                absorption_coeffs=obj.acoustic_shader.get('absorption', {1000: 0.1}),
                                reflection_coeffs=obj.acoustic_shader.get('reflection', {1000: 0.9}),
                                scattering_coeffs=obj.acoustic_shader.get('scattering', {1000: 0.05})
                            )
        except Exception as e:
            print(f"Error updating object {obj.name}: {e}")
    
    def _update_source_voxels(self, source: Any, current_frame: int):
        """Update voxels occupied by sources"""
        try:
            if source.idx in self.source_positions:
                pos_data = self.source_positions[source.idx]
                if current_frame < len(pos_data):
                    position = pos_data[current_frame, :3]
                    
                    voxel_pos = self._world_to_voxel(position)
                    
                    if self._is_in_bounds(voxel_pos):
                        i, j, k = voxel_pos
                        
                        audio_sample = 0.0
                        if source.idx in self.source_audio:
                            audio_data = self.source_audio[source.idx]
                            if current_frame < len(audio_data):
                                audio_sample = audio_data[current_frame]
                        
                        self.grid[i, j, k] = Soxel(
                            idx=source.idx,
                            is_source=True,
                            sound_speed=source.acoustic_shader.get('sound_speed', 343.0),
                            density=source.acoustic_shader.get('density', 1.2),
                            audio_sample=audio_sample
                        )
        except Exception as e e:
            print(f"Error updating source {source.idx}: {e}")
    
    def _voxelize_mesh(self, mesh: trimesh.Trimesh) -> np.ndarray:
        """Voxelize a 3D mesh into the grid"""
        try:
            # Use trimesh voxelization if available
            if hasattr(trimesh, 'voxel'):
                voxelized = mesh.voxelized(self.voxel_size)
                voxels = voxelized.fill()
                return voxels
            else:
                # Fallback to simple bounding box check
                return self._simple_voxelization(mesh)
        except Exception:
            return self._simple_voxelization(mesh)
    
    def _simple_voxelization(self, mesh: trimesh.Trimesh) -> np.ndarray:
        """Simple voxelization using bounding box and point containment"""
        voxels = np.zeros(self.shape, dtype=bool)
        bounds = mesh.bounds
        
        # Calculate voxel range
        min_voxel = self._world_to_voxel(bounds[0])
        max_voxel = self._world_to_voxel(bounds[1])
        
        # Check each voxel in the bounding box
        for i in range(max(0, int(min_voxel[0])), min(self.shape[0], int(max_voxel[0]) + 1)):
            for j in range(max(0, int(min_voxel[1])), min(self.shape[1], int(max_voxel[1]) + 1)):
                for k in range(max(0, int(min_voxel[2])), min(self.shape[2], int(max_voxel[2]) + 1)):
                    world_pos = self._voxel_to_world((i, j, k))
                    if mesh.contains([world_pos]):
                        voxels[i, j, k] = True
        
        return voxels
    
    def _world_to_voxel(self, world_pos: np.ndarray) -> Tuple[int[int, int, int]:
        """Convert world coordinates to voxel indices"""
        return (
            int(world_pos[0] / self.voxel_size),
            int(world_pos[1] / self.voxel_size),
            int(world_pos[2] / self.voxel_size)
        )
    
    def _voxel_to_world(self, voxel_pos: Tuple[int, int, int]) -> np.ndarray:
        """Convert voxel indices to world coordinates"""
        return np.array([
            voxel_pos[0] * self.voxel_size,
            voxel_pos[1] * self.voxel_size,
            voxel_pos[2] * self.voxel_size
        ])
    
    def _is_in_bounds(self, voxel_pos: Tuple[int, int, int]) -> bool:
        """Check if voxel coordinates are within grid bounds"""
        i, j, k = voxel_pos
        return (0 <= i < self.shape[0] and 
                0 <= j < self.shape[1] and 
                0 <= k < self.shape[2])
    
    def get_voxel_center(self, voxel_pos: Tuple[int, int, int]) -> np.ndarray:
        """Get world coordinates of voxel center"""
        return self._voxel_to_world(voxel_pos) + self.voxel_size / 2
    
    def find_nearest_voxel(self, world_pos: np.ndarray) -> Tuple[int, int, int]:
        """Find the nearest voxel to world coordinates"""
        return self._world_to_voxel(world_pos)

