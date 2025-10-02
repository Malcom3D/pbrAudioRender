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

# sources/wave_source.py
import numpy as np
import numba as nb
from numba import float32, int32
from typing import Tuple, Optional

class WaveSource:
    """Base class for wave sources"""
    
    def __init__(self, position: Tuple[float, float, float], amplitude: float = 1.0):
        self.position = position
        self.amplitude = amplitude
        self.active = True

class SphericalWaveSource(WaveSource):
    """Spherical wave source (monopole)"""
    
    def __init__(self, position: Tuple[float, float, float], amplitude: float = 1.0, frequency: Optional[float] = None, audio_data: Optional[np.ndarray] = None):
        super().__init__(position, amplitude)
        self.frequency = frequency
        self.audio_data = audio_data
        self.current_sample = 0
    
    def get_pressure(self, t: float, dt: float) -> float:
        """Get pressure value at time t"""
        if self.audio_data is not None:
            # Use provided audio data
            if self.current_sample < len(self.audio_data):
                print('SphericalWaveSource.get_pressure curr_samp, len: ', self.current_sample, len(self.audio_data))
                pressure = self.audio_data[self.current_sample] * self.amplitude
                self.current_sample += 1
                return pressure
            return 0.0
        elif self.frequency is not None:
            # Generate sine wave
            return self.amplitude * np.sin(2 * np.pi * self.frequency * t)
        else:
            # Impulse
            return self.amplitude if t < dt else 0.0

class PlaneWaveSource(WaveSource):
    """Plane wave source"""
    
    def __init__(self, position: Tuple[float, float, float], direction: Tuple[float, float, float],
                 amplitude: float = 1.0, frequency: Optional[float] = None,
                 audio_data: Optional[np.ndarray] = None):
        super().__init__(position, amplitude)
        self.direction = np.array(direction) / np.linalg.norm(direction)
        self.frequency = frequency
        self.audio_data = audio_data
        self.current_sample = 0
    
    def get_pressure(self, t: float, dt: float) -> float:
        """Get pressure value at time t"""
        if self.audio_data is not None:
            if self.current_sample < len(self.audio_data):
                pressure = self.audio_data[self.current_sample] * self.amplitude
                self.current_sample += 1
                return pressure
            return 0.0
        elif self.frequency is not None:
            return self.amplitude * np.sin(2 * np.pi * self.frequency * t)
        else:
            return self.amplitude if t < dt else 0.0

class ModalSource(WaveSource):
    """Modal synthesis source for impact sounds"""
    
    def __init__(self, position: Tuple[float, float, float], 
                 material_properties: dict, geometry: np.ndarray):
        super().__init__(position, 1.0)
        self.material_properties = material_properties
        self.geometry = geometry
        self.modal_data = None
        self.impact_force = None
    
    def compute_modes(self):
        """Compute vibration modes for the geometry"""
        # This would use the ModalSynthesizer from modal_synthesis.py
        pass
    
    def apply_impact(self, force: np.ndarray, location: Tuple[float, float, float]):
        """Apply impact force at specific location"""
        self.impact_force = force
        self.impact_location = location
