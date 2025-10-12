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

"""
Solid medium shader with frequency-dependent acoustic properties.
Implements elastic solid media with complex modulus, damping, and anisotropy.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json

from ..lib.frequency_response import MaterialFrequencyResponse


@dataclass
class SolidShader:
    """Acoustic shader for solid media (metals, plastics, wood, etc.)"""
    
    name: str = "steel"
    youngs_modulus: float = 200e9  # Young's modulus in Pa
    shear_modulus: float = 80e9    # Shear modulus in Pa
    density: float = 7850.0        # Density in kg/m³
    poissons_ratio: float = 0.3    # Poisson's ratio
    loss_factor: float = 0.001     # Material loss factor (damping)
    temperature: float = 20.0      # Celsius
    anisotropy: Dict[str, float] = None  # Anisotropic properties
    frequency_response: MaterialFrequencyResponse = None
    
    def __post_init__(self):
        if self.anisotropy is None:
            self.anisotropy = {'E1': self.youngs_modulus, 'E2': self.youngs_modulus, 'E3': self.youngs_modulus}
        
        if self.frequency_response is None:
            self.frequency_response = MaterialFrequencyResponse()
            self._initialize_default_response()
    
    def _initialize_default_response(self):
        """Initialize default frequency response for solid"""
        frequencies = np.array([100, 500, 1000, 5000, 10000, 20000, 50000], dtype=float)
        absorption = self.calculate_solid_absorption(frequencies)
        
        self.frequency_response.frequencies = frequencies
        self.frequency_response.absorption_coeffs = absorption
    
    def calculate_longitudinal_sound_speed(self) -> float:
        """
        Calculate longitudinal wave speed in solid.
        
        Returns:
            Longitudinal wave speed in m/s
        """
        # For isotropic materials
        E = self.youngs_modulus
        ν = self.poissons_ratio
        ρ = self.density
        
        # Longitudinal wave speed: c_L = √((λ + 2μ)/ρ)
        # where λ = Eν/((1+ν)(1-2ν)) and μ = E/(2(1+ν))
        λ = (E * ν) / ((1 + ν) * (1 - 2 * ν))
        μ = E / (2 * (1 + ν))
        
        c_L = np.sqrt((λ + 2 * μ) / ρ)
        
        return float(c_L)
    
    def calculate_shear_sound_speed(self) -> float:
        """
        Calculate shear wave speed in solid.
        
        Returns:
            Shear wave speed in m/s
        """
        E = self.youngs_modulus
        ν = self.poissons_ratio
        ρ = self.density
        
        # Shear modulus
        μ = E / (2 * (1 + ν))
        
        # Shear wave speed: c_S = √(μ/ρ)
        c_S = np.sqrt(μ / ρ)
        
        return float(c_S)
    
    def calculate_rayleigh_sound_speed(self) -> float:
        """
        Calculate Rayleigh surface wave speed.
        
        Returns:
            Rayleigh wave speed in m/s
        """
        c_L = self.calculate_long_longitudinal_sound_speed()
        c_S = self.calculate_shear_sound_speed()
        
        # Approximate Rayleigh wave speed (exact solution requires solving cubic)
        # Good approximation: c_R ≈ (0.87 + 1.12ν)/(1 + ν) * c_S
        ν = self.poissons_ratio
               c_R = ((0.87 + 1.12 * ν) / (1 + ν)) * c_S
        
        return float(c_R)
    
    def calculate_density(self) -> float:
        """
        Get material density.
        
        Returns:
            Density in kg/m³
        """
        return self.density
    
    def calculate_solid_absorption(self, frequencies: np.ndarray) -> np.ndarray:
        """
        Calculate absorption in solid media.
        
        Args:
            frequencies: Array of frequencies in Hz
        
        Returns:
            Array of absorption coefficients in Np/m
        """
        # For solids, absorption often follows a power law: α = α₀ * f^β
        # where β is typically between 1 and 2
        
        # Base absorption coefficient at 1 kHz
        alpha_1k = self.loss_factor * np.pi * 1000 / self.calculate_longitudinal_sound_speed()
        
        # Frequency exponent (material dependent)
        beta = 1.5  # Typical for many solids
        
        # Calculate absorption
        alpha = alpha_1k * (frequencies / 1000)**beta
        
        return alpha
    
    def get_complex_modulus(self, frequency: float) -> complex:
        """
        Get complex Young's modulus accounting for material damping.
        
        Args:
            frequency: Frequency in Hz
        
        Returns:
            Complex Young's modulus in Pa
        """
        # Real part (storage modulus)
        E_real = self.youngs_modulus
        
        # Imaginary part (loss modulus)
        E_imag = self.youngs_modulus * self.loss_factor
        
        return complex(E_real, E_imag)
    
    def get_absorption_at_frequency(self, frequency: float) -> float:
        """Get absorption coefficient at specific frequency"""
        return self.frequency_response.get_absorption_at_frequency(frequency)
    
    def get_impedance_at_frequency(self, frequency: float) -> complex:
        """Get characteristic impedance at specific frequency"""
        sound_speed = self.calculate_longitudinal_sound_speed()
        density = self.density
        
        # For solids with damping, impedance is complex
        complex_E = self.get_complex_modulus(frequency)
        
        # Complex wave number
        k_real = 2 * np.pi * frequency / sound_speed
        k_imag = self.get_absorption_at_frequency(frequency)
        k_complex = complex(k_real, k_imag)
        
        # Complex impedance: Z = ρ * ω / k
        omega = 2 * np.pi * frequency
        impedance = density * omega / k_complex
        
        return impedance
    
    def calculate_reflection_coefficient(self, other_impedance: complex, 
                                       angle: float = 0.0) -> complex:
        """
        Calculate reflection coefficient at interface with another medium.
        
        Args:
            other_impedance: Impedance of other medium
            angle: Incidence angle in radians (0 = normal incidence)
        
        Returns:
            Complex reflection coefficient
        """
        Z1 = self.get_impedance_at_frequency(1000)  # Use 1kHz as reference
        Z2 = other_impedance
        
        if angle == 0:
            # Normal incidence
            R = (Z2 - Z1) / (Z2 + Z1)
        else:
            # Oblique incidence (simplified)
            # This would need Snell's law and mode conversion for solids
            Z1_eff = Z1 / np.cos(angle)
            Z2_eff = Z2 / np.cos(self._calculate_transmission_angle(angle, Z1, Z2))
            R = (Z2_eff - Z1_eff) / (Z2_eff + Z1_eff)
        
        return R
    
    def _calculate_transmission_angle(self, incidence_angle: float, 
                                   Z1: complex, Z2: complex) -> float:
        """Calculate transmission angle using Snell's law"""
        c1 = self.calculate_longitudinal_sound_speed()
        
        # Assume other medium has similar wave speed for simplicity
        # In practice, you'd need the other medium's properties
        c2 = c1 * 0.8  # Approximation
        
        sin_transmission = (c2 / c1) * np.sin(incidence_angle)
        transmission_angle = np.arcsin(min(1.0, abs(sin_transmission)))
        
        return transmission_angle
    
    def to_soxel_properties(self, frequency: float = 1000.0) -> Dict[str, Any]:
        """Convert to Soxel-compatible properties dictionary"""
        return {
            'sound_speed': self.calculate_longitudinal_sound_speed(),
            'density': self.density,
            'absorption_coeffs': {frequency: self.get_absorption_at_frequency(frequency)},
            'reflection_coeffs': {frequency: abs(self.calculate_reflection_coefficient(complex(400), 0))},
            'shear_sound_speed': self.calculate_shear_sound_speed(),
            'rayleigh_sound_speed': self.calculate_rayleigh_sound_speed(),
            'impedance': self.get_impedance_at_frequency(frequency),
            'shader_type': 'solid',
            'name': self.name
        }
    
    @classmethod
    def from_preset(cls, preset_name: str) -> 'SolidShader':
        """Create solid shader from common presets"""
        presets = {
            'steel': {
                'name': 'steel',
                'youngs_modulus': 200e9,
                'density': 7850.0,
                'poissons_ratio': 0.3,
                'loss_factor': 0.001
            },
            'aluminum': {
                'name': 'aluminum',
                'youngs_modulus': 70e9,
                'density': 2700.0,
                'poissons_ratio': 0.33,
                'loss_factor': 0.001
            },
            'glass': {
                'name': 'glass',
                'youngs_modulus': 70e9,
                'density': 2500.0,
                'poissons_ratio': 0.22,
                'loss_factor': 0.002
            },
            'concrete': {
                'name': 'concrete',
                'youngs_modulus': 30e9,
                'density': 2400.0,
                'poissons_ratio': 0.2,
                'loss_factor': 0.01
            },
            'wood_oak': {
                'name': 'wood_oak',
                'youngs_modulus': 11e9,
                'density': 700.0,
                'poissons_ratio': 0.3,
                'loss_factor': 0.02
            },
            'rubber': {
                'name': 'rubber',
                'youngs_modulus': 0.01e9,
                'density': 1100.0,
                'poissons_ratio': 0.49,
                'loss_factor': 0.1
            },
            'foam': {
                'name': 'foam',
                'youngs_modulus': 0.001e9,
                'density': 50.0,
                'poissons_ratio': 0.4,
                'loss_factor': 0.2
            }
        }
        
        if preset_name in presets:
            return cls(**presets[preset_name])
        else:
            raise ValueError(f"Unknown solid preset: {preset_name}")
    
    def to_dict(self) -> Dict:
        """Convert solid shader to dictionary"""
        return {
            'name': self.name,
            'youngs_modulus': self.youngs_modulus,
            'shear_modulus': self.shear_modulus,
            'density': self.density,
            'poissons_ratio': self.poissons_ratio,
            'loss_factor': self.loss_factor,
            'temperature': self.temperature,
            'anisotropy': self.anisotropy,
            'frequency_response': self.frequency_response.to_dict() if self.frequency_response else None
        }
    
    def save_to_file(self, file_path: str):
        """Save solid shader to JSON file"""
        data = self.to_dict()
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load_from_file(cls, file_path: str) -> -> 'SolidShader':
        """Load solid shader from JSON file"""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Handle frequency response
        fr_data = data.pop('frequency_response', None)
        shader = cls(**data)
        
        if fr_data and shader.frequency_response:
            shader.frequency_response.load_from_dict(fr_data)
        
        return shader

