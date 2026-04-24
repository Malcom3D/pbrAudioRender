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
from ..core.entity_manager import EntityManager
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

@dataclass
class AcousticScene:
    """SIMD-friendly frequency bands aware data structure for 3D scene info"""
    freq_bands: List[Tuple[float, float]]

    def __post_init__(self):
        # Number of frequency bands
        n_bands = len(self.freq_bands)
        self.mesh_info = np.zeros((0,3,3), dtype=np.float32)
        self.scene_info = np.array([], dtype=np.int32)
        self.mat_info = {}

        # Init store for acoustiic material info
        self.sound_speed = np.zeros((0,1), dtype=np.float32)
        self.density = np.zeros((0,1), dtype=np.float32)
        self.absorption = np.zeros((0,2,n_bands), dtype=np.float32)
        self.refraction = np.zeros((0,2,n_bands), dtype=np.float32)
        self.reflection = np.zeros((0,2,n_bands), dtype=np.float32)
        self.scattering = np.zeros((0,2,n_bands), dtype=np.float32)

        # Init ASO store for acoustic source and output
        num_aso = 2
        self.aso_pos = np.zeros((num_aso, 3), dtype=np.float32)
        self.aso_medium = np.zeros(num_aso, dtype=np.int32)
        self.aso_radius = np.empty(num_aso, dtype=np.float32)

    def add_aso_info(self, aso_id: int, position: np.ndarray, medium_idx: int, src_radius: float = None):
        idx = 0 if aso_id == -2 else 1
        self.aso_pos[idx] = position.tolist()
        self.aso_medium[idx] = medium_idx
        self.aso_radius[idx] = radius if not radius == None else np.nan

    def add_mesh_info(self, obj_idx: int, obj_config: Any, vertices: np.ndarray, faces: np.ndarray):
        n_bands = len(self.freq_bands)
        # Get triangle count
        triangle_count = faces.shape[0]

        self.scene_info = np.append(self.scene_info, np.full((triangle_count,), obj_idx, dtype=np.int32))
        self.mesh_info = np.append(self.mesh_info, vertices[faces], axis=0)

        sound_speed = obj_config.acoustic_shader.sound_speed
        density = obj_config.acoustic_shader.density
        self.sound_speed = np.append(self.sound_speed, np.full((triangle_count,), sound_speed, dtype=np.float32))
        self.density = np.append(self.density, np.full((triangle_count,), density, dtype=np.float32))

        # Get Material Info
        if obj_idx >= -1:
            self.ac_sound_speed = obj_config.acoustic_shader.sound_speed
            self.ac_density = obj_config.acoustic_shader.density

            # get acoustic domain absorption coefficients and phases and save as main medium info
            temperature = obj_config.acoustic_shader.temperature
            impedence = obj_config.acoustic_shader.impedence
            coeffs, phases = _compute_acoustic_coefficients(sound_speed, density, temperature, impedence, np.unique(self.freq_bands)[:-1])
            self.ac_absorption = np.append(np.vstack(coeffs), np.vstack(phases), axis=1).astype(np.float32)

            # Complete array with null value
            coeffs = [1 for _ in range(len(self.freq_bands))]
            phases = [0 for _ in range(len(self.freq_bands))]
            self.absorption = np.append(self.absorption, np.full((triangle_count,2,n_bands), [coeffs, phases], dtype=np.float32))
            self.refraction = np.append(self.refraction, np.full((triangle_count,2,n_bands), [coeffs, phases], dtype=np.float32))
            self.reflection = np.append(self.reflection, np.full((triangle_count,2,n_bands), [coeffs, phases], dtype=np.float32))
            self.scattering = np.append(self.scattering, np.full((triangle_count,2,n_bands), [coeffs, phases], dtype=np.float32))

        if obj_idx >= 0:
            # Get Object AcousticShader
            coeffs, phases = obj_config.acoustic_shader.acoustic_properties.absorption.get_bands_avg(self.freq_bands)
            coeffs = coeffs.tolist()
            phases = phases.tolist() if not phases == None else [0 for _ in range(len(self.freq_bands))]
            self.absorption = np.append(self.absorption, np.full((triangle_count,2,n_bands), [coeffs, phases], dtype=np.float32))

            coeffs, phases = obj_config.acoustic_shader.acoustic_properties.refraction.get_bands_avg(self.freq_bands)
            coeffs = coeffs.tolist()
            phases = phases.tolist() if not phases == None else [0 for _ in range(len(self.freq_bands))]
            self.refraction = np.append(self.refraction, np.full((triangle_count,2,n_bands), [coeffs, phases], dtype=np.float32))

            coeffs, phases = obj_config.acoustic_shader.acoustic_properties.reflection.get_bands_avg(self.freq_bands)
            coeffs = coeffs.tolist()
            phases = phases.tolist() if not phases == None else [0 for _ in range(len(self.freq_bands))]
            self.reflection = np.append(self.reflection, np.full((triangle_count,2,n_bands), [coeffs, phases], dtype=np.float32))

            coeffs, phases = obj_config.acoustic_shader.acoustic_properties.scattering.get_bands_avg(self.freq_bands)
            coeffs = coeffs.tolist()
            phases = phases.tolist() if not phases == None else [0 for _ in range(len(self.freq_bands))]
            self.scattering = np.append(self.scattering, np.full((triangle_count,2,n_bands), [coeffs, phases], dtype=np.float32))

    def _compute_acoustic_coefficients(c: float, rho: float, T: float, Z: float, freqs: np.ndarray):
        """
        Compute absorption coefficient and phase shift coefficient for air.
    
        Parameters:
        -----------
        c : float
            Speed of sound in air (m/s)
        rho : float
            Density of of air (kg/m³)
        T : float
            Temperature in °C
        Z : float
            Characteristic impedance of air (rayls)
        freqs : array
            Frequency (Hz) - bands frequency
        Returns:
        --------
        alpha : array
            Absorption coefficient (nepers/m)
        beta : array
            Phase shift coefficient (rad/m)
        """

        # Convert temperature to Kelvin
        T_K = T + 273.15

        # Compute Angular frequency
        omega = 2 * np.pi * freqs

        # Stokes-Kirchhoff formula for sound absorption
        # Constants for air
        mu = 1.846e-5  # Dynamic viscosity (Pa·s) at 20°C
        kappa = 0.0262  # Thermal conductivity (W/m·K) at 20°C
        Cp = 1005  # Specific heat at constant pressure (J/kg·K)
        Cv = 718  # Specific heat at constant volume (J/kg·K)
        gamma_specific = Cp / Cv  # Ratio of specific heats

        
        # Viscous contribution
        alpha_visc = (omega**2 * mu) / (2 * rho * c**3)
        
        # Thermal contribution
        alpha_therm = (omega**2 * kappa * (gamma_specific - 1)) / (2 * rho * c**3 * Cp)
        
        # Total absorption coefficient
        alpha = alpha_visc + alpha_therm
        
        # Phase shift coefficient
        beta = omega / c

        return alpha, beta
