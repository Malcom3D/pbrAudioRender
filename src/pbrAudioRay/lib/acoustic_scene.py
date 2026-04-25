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

from .functions import _compute_rayleigh_damping

@dataclass
class AcousticScene:
    """SIMD-friendly frequency bands aware data structure for 3D scene info"""
    freq_bands: List[Tuple[float, float]]
    num_objs: int = 0

    def __post_init__(self):
        # Number of frequency bands
        n_bands = len(self.freq_bands)
        self.mesh_info = np.zeros((0,3,3), dtype=np.float32)
        self.scene_info = np.array([], dtype=np.int32)

        # Init store for acoustiic material info
        self.sound_speed = np.zeros((0,1), dtype=np.float32)
        self.density = np.zeros((0,1), dtype=np.float32)
        self.roughness = np.zeros((0,1), dtype=np.float32)
        self.absorption = np.zeros((0,2,n_bands), dtype=np.float32)
        self.refraction = np.zeros((0,2,n_bands), dtype=np.float32)
        self.reflection = np.zeros((0,2,n_bands), dtype=np.float32)
        self.scattering = np.zeros((0,2,n_bands), dtype=np.float32)

        # Init ASO store for acoustic source and output
        num_aso = 2
        self.aso_pos = np.zeros((num_aso, 3), dtype=np.float32)
        self.aso_medium = np.zeros(num_aso, dtype=np.int32)
        self.aso_radius = np.empty(num_aso, dtype=np.float32)

    def set_num_objs(self, num_objs: int):
        n_bands = len(self.freq_bands)
        self.objs_idx = np.zeros((num_objs,1), dtype=np.int32)
        self.objs_medium = np.zeros((num_objs,2,n_bands), dtype=np.float32)

    def add_aso_info(self, aso_id: int, position: np.ndarray, medium_idx: int, src_radius: float = None):
        idx = 0 if aso_id == -2 else 1
        self.aso_pos[idx] = position.tolist()
        self.aso_medium[idx] = medium_idx
        self.aso_radius[idx] = radius if not radius == None else np.nan

    def add_mesh_info(self, obj_idx: int, obj_config: Any, vertices: np.ndarray, faces: np.ndarray):

        # register obj_idx
        self.objs_idx[self.num_objs] = obj_idx

        n_bands = len(self.freq_bands)
        # Get triangle count
        triangle_count = faces.shape[0]

        self.scene_info = np.append(self.scene_info, np.full((triangle_count,), obj_idx, dtype=np.int32))
        self.mesh_info = np.append(self.mesh_info, vertices[faces], axis=0)

        # Get Material Info
        if obj_idx == -1:
            # Get AcousticDomain AcousticShader and save as main medium info
            sound_speed = obj_config.acoustic_shader.sound_speed
            temperature = obj_config.acoustic_shader.temperature
            impedence = obj_config.acoustic_shader.impedence
            density = obj_config.acoustic_shader.density

            self.ac_sound_speed = sound_speed
            self.ac_density = density

            # Compute acoustic domain attenuation coefficients and phases and save as medium info
            coeffs, phases = self._compute_acoustic_domain_coefficients(c=sound_speed, rho=density, T=temperature, Z=impedence)
            self.objs_medium[self.num_objs] = [coeffs.reshape(n_bands,), phases.reshape(n_bands,)]
            self.ac_attenuation = np.append(np.vstack(coeffs), np.vstack(phases), axis=1).astype(np.float32)

            # Add AcousticDomain AcousticShader data to material info
            self.sound_speed = np.append(self.sound_speed, np.full((triangle_count,), sound_speed, dtype=np.float32))
            self.density = np.append(self.density, np.full((triangle_count,), density, dtype=np.float32))
            self.roughness = np.append(self.roughness, np.full((triangle_count,), [-1], dtype=np.float32))

            # Complete material info array with null value
            coeffs = [1 for _ in range(len(self.freq_bands))]
            phases = [0 for _ in range(len(self.freq_bands))]
            self.absorption = np.append(self.absorption, np.full((triangle_count,2,n_bands), [coeffs, phases], dtype=np.float32))
            self.refraction = np.append(self.refraction, np.full((triangle_count,2,n_bands), [coeffs, phases], dtype=np.float32))
            self.reflection = np.append(self.reflection, np.full((triangle_count,2,n_bands), [coeffs, phases], dtype=np.float32))
            self.scattering = np.append(self.scattering, np.full((triangle_count,2,n_bands), [coeffs, phases], dtype=np.float32))

        if obj_idx >= 0:
            # Get Object AcousticShader and save in material info array
            sound_speed = obj_config.acoustic_shader.sound_speed
            self.sound_speed = np.append(self.sound_speed, np.full((triangle_count,), sound_speed, dtype=np.float32))

            density = obj_config.acoustic_shader.density
            self.density = np.append(self.density, np.full((triangle_count,), density, dtype=np.float32))

            roughness = obj_config.acoustic_shader.roughness
            self.roughness = np.append(self.roughness, np.full((triangle_count,), roughness, dtype=np.float32))

            young_modulus = obj_config.acoustic_shader.young_modulus
            poisson_ratio = obj_config.acoustic_shader.poisson_ratio
            damping = obj_config.acoustic_shader.damping
            for idx in range(n_bands):
                min_freq, max_freq = self.freq_bands[idx]
                coeffs, phases = self._compute_acoustic_object_coefficients(sound_speed, density, young_modulus, poisson_ratio, damping)
                self.objs_medium[self.num_objs] = [coeffs.reshape(n_bands,), phases.reshape(n_bands,)]

            # Get Object AcousticProperties
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

        self.num_objs += 1

    def _compute_acoustic_object_coefficients(self, c: float, rho: float, E: float, nu: float, damping: float):
            """
            Calculate medium attenuation coefficient and phase shift for acoustic objects.
            Works for gases, fluids, and solids using a simplified common method.
            """
            # Compute Rayleigh damping coefficient α (mass proportional) and Rayleigh damping coefficient β (stiffness proportional)
            n_bands = len(self.freq_bands)
            alpha = beta = np.zeros((n_bands,1), dtype=np.float32)
            for idx in range(n_bands):
                min_freq, max_freq = self.freq_bands[idx]
                alpha[idx], beta[idx] = _compute_rayleigh_damping(min_freq, max_freq, damping)

            freqs = np.unique(self.freq_bands)[:-1]
            omega = 2 * np.pi * freqs
            omega = omega.reshape(freqs.shape[0],1)

            # Calculate derived properties from input parameters.
            # Bulk modulus (for fluids and isotropic solids)
            K = E / (3 * (1 - 2 * nu))
        
            # Shear modulus (for solids)
            G = E / (2 * (1 + nu))
        
            # Characteristic impedance
            Z = rho * c
        
            # Determine if it's likely a solid (has significant shear modulus)
            is_solid = G > 0.1 * E

            # Compute attenuation coefficient α_att (in Np/m) using Rayleigh damping model: α_att = (α / (2*c)) + (β * ω² / (2*c))
            alpha_attenuation = (alpha / (2 * c)) + (beta * omega**2 / (2 * c))

            # For fluids and gases add a simple viscous term
            if not is_solid:
                # Simplified viscous attenuation (Stokes' law approximation)
                # Assuming dynamic viscosity ≈ 1.8e-5 Pa·s for air, 1e-3 Pa·s for water
                viscosity = 1.8e-5 if rho < 100 else 1e-3  # rough approximation
                alpha_viscous = (2 * omega**2 * viscosity) / (3 * rho * c**3)
            else:
                alpha_viscous = 0
        
            # Total attenuation coefficient
            alpha_attenuation = alpha_attenuation + alpha_viscous

            # Compute phase shift (in rads/m)
            phase_shift = omega / c

            return alpha_attenuation, phase_shift

    def _compute_acoustic_domain_coefficients(self, c: float, rho: float, T: float, Z: float):
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

        freqs = np.unique(self.freq_bands)[:-1]

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
