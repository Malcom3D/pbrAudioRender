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
import numba as nb
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

from ...core.entity_manager import EntityManager

@dataclass
class AbsorptionInterface:
    entity_manager: EntityManager
    acoustic_scene: Any  # AcousticScene

#    def compute_attenuation(self, origins: np.ndarray, hit_points: np.ndarray, medium: np.ndarray, bands_idx: int, ray_data: Any):
    def compute_attenuation(self, origins: np.ndarray, hit_points: np.ndarray, bands_idx: int, ray_data: Any):
        """Apply frequency-dependent medium attenuation."""

        # Get main medium properties
        ac_sound_speed = self.acoustic_scene.ac_sound_speed
        ac_density = self.acoustic_scene.ac_density
        ac_attenuation = self.acoustic_scene.ac_attenuation[bands_idx]

        # Compute traveled path length
        path_length = np.sqrt(np.sum((hit_points - origins)**2, axis=1)).reshape(-1,1)
        ray_data.path_length = path_length

        # Compute dalay in main medium
        delay = path_length * ac_sound_speed # all path are on the acoustic domain
        ray_data.delay = delay

        # Compute energy attenuation and phase shift after traveled path using exponential decay
        # E = E0 * exp(-alpha * distance)
        # where alpha is in nepers/m
        initial_energy = ray_data.energies
        attenuation = np.exp(-ac_attenuation[0] * path_length)
        rays_energies = initial_energy * attenuation

        # Calculate phase shift
        # Phase = beta * distance (in radians)
        initial_phase = ray_data.phases
        phase_shift = ac_attenuation[1] * path_length
        rays_phases = (initial_phase + phase_shift) % (2 * np.pi)

        # Wrap phase to [-π, π] range for better numerical representation
        rays_phases = np.mod(rays_phases + np.pi, 2 * np.pi) - np.pi

        rays_energies = rays_energies.reshape(-1,1)
        rays_phases = rays_phases.reshape(-1,1)

        return rays_energies, rays_phases

    def compute(self, energies: np.ndarray, phases: np.ndarray, incident_angles: np.ndarray, absorption_coeffs: np.ndarray, absorption_phases: np.ndarray, bands_idx: int, ray_data: Any):
        """Apply frequency-dependent intersections absorption."""

        print('absorbed_energy', absorbed_energy.shape, rays_energies.shape, rays_phases.shape, incident_angles.shape, absorption_coeffs.shape)

        # Compute intersection absorption energies (no phase shift)
        angle_factor = np.cos(incident_angles)
        angle_factor[angle_factor == 0] = 1e-10
        absorbed_energy = energies * absorption_coeffs * angle_factor

        absorbed_energy = absorbed_energy.reshape(-1,1)
        phases = phases.reshape(-1,1)

        return absorbed_energy, phases
