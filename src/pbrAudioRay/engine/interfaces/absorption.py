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
from ...lib.acoustic_shader import AcousticShader
from ...lib.ray_data import RayData

@dataclass
class AbsorptionInterface:
    entity_manager: EntityManager
    
    def compute(self, ray: RayData):
        """Apply frequency-dependent absorption."""
        # get fraquency bands
        frequency_bands = self.entity_manager.get('frequency_bands')
        freq_bands = frequency_bands.get_bands()

        # Get ray data
        low_freq, high_freq = freq_bands[ray.bands_idx]
        shader = ray.medium.acoustic_shader

        # Absorption: reduce energy
        if ray.medium.type == 'world':
            print(low_freq, high_freq, ray.length, shader)
            coeff, phase = self.get_ac_avg_coeffs(low_freq, high_freq, ray.length, shader)
            ray.energy *= (1 - coeff)
            if not phase == None:
                ray.phase = np.angle(np.exp(1j * (ray.phase + phase)))

        elif hasattr(shader, 'acoustic_properties') and hasattr(shader.acoustic_properties, 'absorption'):
            coeff, phase = shader.acoustic_properties.absorption.get_avg_coeffs(low_freq, high_freq)
            ray.energy *= (1 - coeff)
            if not phase == None:
                ray.phase = np.angle(np.exp(1j * (ray.phase + phase)))

        return ray

    def get_ac_avg_coeffs(low_freq: float, high_freq: float, distance: float, shader: AcousticShader, humidity: float = 50.0) -> Dict[str, np.ndarray]:
        """
        Compute frequency-dependent attenuation coefficients and spatial phase shift for a spherical sound source.
    
        Parameters:
        -----------
        low_freq : float
            Lowest frequency of interest [Hz]
        high_freq : float
            Highest frequency of interest [Hz]
        distance : float
            Distance from source [m]
        shader: AcousticShader
            AcousticShader of the AcousticDomain
        humidity : float
            Relative humidity [%] for air absorption calculation (default: 50)
    
        Returns:
        --------
        dict containing:
            - coeff: Total attenuation coefficient [1/m]
            - phase_shift_wrapped: Wrapped spatial phase shift [rad]
        """
    
        # Generate logarithmically spaced frequencies points for average
        print(low_freq, high_freq, num=100, endpoint=True)
        frequency = np.geomspace(low_freq, high_freq, num=100, endpoint=True)

        # Get shader data
        sound_speed = shader.sound_speed
        temperature = shader.temperature
        density = shader.density

        # Physical constants
        T_ref = 273.15  # Reference temperature (20°C) in Kelvin
        p_ref = 101325  # Reference atmospheric pressure [Pa]
        T_kelvin = temperature + T_ref  # Convert to Kelvin

        # Sutherland's formula for air viscosity
        S = 110.4  # Sutherland constant for air in Kelvin
        mu_ref = 1.716e-5  # Reference viscosity at 273.15K
        viscosity = mu_ref * (temperature / T_ref)**1.5 * (T_ref + S) / (temperature + S)

        # Approximate thermal conductivity for air
        k_ref = 0.0241  # W/(m·K) at 273.15K
        thermal_conductivity = k_ref * (temperature / T_ref)**0.81
    
        # Angular frequency
        omega = 2 * np.pi * frequency
    
        # Wave number in ideal (lossless) medium
        k_real = omega / sound_speed
    
        # Classical absorption due to viscosity and thermal conductivity
        # Stokes-Kirchhoff attenuation formula
        Pr = viscosity * specific_heat_ratio * 287.05 / thermal_conductivity  # Prandtl number approx
    
        # Classical attenuation coefficient (frequency squared dependence)
        gamma = specific_heat_ratio
        alpha_classical = (omega**2 * viscosity) / (2 * density * sound_speed**3) * (4/3 + (gamma - 1) / Pr)

        # Calculate relaxation frequencies (ISO 9613-1 standard)
        # Oxygen relaxation frequency
        frO = (p_ref / 101325) * (24 + 4.04e4 * humidity * (0.02 + humidity) / (0.391 + humidity))
    
        # Nitrogen relaxation frequency (ISO 9613-1 standard)
        frN = (p_ref / 101325) * (T_kelvin / T_ref)**(-0.5) * (9 + 280 * humidity * np.exp(-4.17 * ((T_kelvin / T_ref)**(-1/3) - 1)))
    
        # Atmospheric absorption coefficient in dB/m (ISO 9613-1 standard)
        alpha_atm_db = np.zeros_like(frequencies)
        for i, f in enumerate(frequencies):
            term1 = 1.84e-11 * (T_kelvin / T_ref)**0.5
            term2 = (T_kelvin / T_ref)**(-2.5)
            term3 = 0.01275 * (np.exp(-2239.1 / T_kelvin)) * (frO / (f**2 + frO**2))
            term4 = 0.1068 * (np.exp(-3352 / T_kelvin)) * (frN / (f**2 + frN**2))
        
            alpha_absorption[i] = f**2 * (term1 + term2 * (term3 + term4))
    
        alpha_total = np.mean(alpha_geometric + alpha_absorption)
        coeff = 1 - np.e**(-alpha_total*distance)
    
        # 4. PHASE SHIFT
        # Wavenumber without attenuation: k = 2πf/c
        k_real = np.mean(2 * np.pi * frequencies / sound_speed)
    
        # Phase shift after distance d: φ = -k_real * d
        phase_shift = -k_real * distance
    
        # Wrap phase to [-π, π] range for convenience
        phase_shift_wrapped = np.angle(np.exp(1j * phase_shift))
    
        return coeff, phase_shift_wrapped
