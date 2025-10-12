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
Fluid medium shader with frequency-dependent acoustic properties.
Implements liquid media with viscosity, temperature, and pressure effects.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json

from ..lib.frequency_response import MaterialFrequencyResponse


@dataclass
class FluidShader:
    """Acoustic shader for fluid media (water, oil, etc.)"""
    
    name: str = "water"
    temperature: float = 20.0  # Celsius
    pressure: float = 101325.0  # Pascals
    salinity: float = 0.0  # For seawater, in ppt
    viscosity: float = 0.001002  # Dynamic viscosity in Pa·s (water at 20°C)
    bulk_modulus: float = 2.15e9  # Bulk modulus in Pa (water)
    density: float = 998.0  # Density in kg/m³ (water at 20°C)
    frequency_response: MaterialFrequencyResponse = None
    
    def __post_init__(self):
        if self.frequency_response is None:
            self.frequency_response = MaterialFrequencyResponse()
            self._initialize_default_response()
    
    def _initialize_default_response(self):
        """Initialize default frequency response for water"""
        frequencies = np.array([100, 500, 1000, 5000,, 10000, 20000, 50000], dtype=float)
        absorption = self.calculate_fluid_absorption(frequencies)
        
        self.frequency_response.frequencies = frequencies
        self.frequency_response.abs.absorption_coeffs = absorption
    
    def calculate_sound_speed(self) -> float:
        """
        Calculate sound speed in fluid.
        
        Returns:
            Sound speed in m/s
        """
        if self.name.lower() == "water":
            return self._calculate_water_sound_speed()
        else:
            # General fluid sound speed: c = √(K/ρ)
            return np.sqrt(self.bulk_modulus / self.density)
    
    def _calculate_water_sound_speed(self) -> float:
        """Calculate sound speed in water using empirical formulas"""
        T = self.temperature  # Celsius
        P = self.pressure / 1e6  # Convert to MPa
        S = self.salinity  # Salinity in ppt
        
        # Coppens equation for sound speed in water
        c0 = 1402.5 + 5.0 * T - 5.44e-2 * T**2 + 2.1e-4 * T**3
        
        # Salinity correction
        c_salinity = 1.33 * S - 1.23e-2 * S * T + 8.7e-5 * S * T**2
        
        # Pressure correction
        c_pressure = 1.56e-1 * P + 2.44e-3 * P**2 - 7.3e-6 * P**3
        
        # Combined sound speed
        sound_speed = c0 + c_salinity + c_pressure
        
        return float(sound_speed)
    
    def calculate_density(self) -> float:
        """
        Calculate fluid density.
        
        Returns:
            Density in kg/m³
        """
        if self.name.lower() == "water":
            return self._calculatecalculate_water_density()
        else:
            return self.density
    
    def _calculate_water_density(self) -> float:
        """Calculate water density considering temperature and salinity"""
        T = self.temperature
        S = self.salinity
        
        # Pure water density at temperature
        rho_0 = 999.842594 + 6.793952e-2 * T - 9.095290e-3 * T**2 + 1.001685e-4 * T**3 - 1.120083e-6 * T**4 + 6.536332e-9 * T**5
        
        # Salinity correction
        A = 0.824493 - 4.0899e-3 * T + 7.6438e-5 * T**2 - 8.2467e-7 * T**3 + 5.3875e-9 * T**4
        B = -5.72466e-3 + 1.0227e-4 * T - 1.6546e-6 * T**2
        C = 4.8314e-4
        
        rho = rho_0 + A * S + B * S**1.5 + C * S**2
        
        return float(rho)
    
    def calculate_fluid_absorption(self, frequencies: np.ndarray) -> np.ndarray:
        """
        Calculate absorption in fluid media.
        
        Args:
            frequencies: Array of frequencies in Hz
        
        Returns:
            Array of absorption coefficients in Np/m
        """
        if self.name.lower() == "water":
            return self._calculate_water_absorption(frequencies)
        else:
            # General fluid absorption: α = (ω² * η) / (2 * ρ * c³)
            omega = 2 * np.pi * frequencies
            c = self.calculate_sound_speed()
            rho = self.calculate_density()
            
            alpha = (omega**2 * self.viscosity) / (2 * rho * c**3)
            return alpha
    
    def _calculate_water_absorption(self, frequencies: np.ndarray) -> np.ndarray:
        """Calculate absorption in water using Francois-Garrison equation"""
        T = self.temperature
        S = self.salinity
        P = self.pressure / 1e6  # MPa
        f = frequencies / 1000.0  # Convert to kHz
        
        # Relaxation frequencies for boric acid and magnesium sulfate
        f1 = 0.78 * np.sqrt(S / 35) * np.exp(T / 26)
        f2 = 42 * np.exp(T / 17)
        
        # Absorption contributions
        A1 = 0.106 * (f1 * f**2) / (f1**2 + f**2) * np.exp((T - 20) / 26)
        A2 = 0.52 * (1 + T / 43) * (S / 35) * (f2 * f**2) / (f2**2 + f**2) * np.exp(-T / 17)
        A3 = 0.00049 * f**2 * np.exp(-(T / 27 + P / 170))
        
        # Total absorption in dB/km, convert to Np/m
        alpha_dB_km = A1 + A2 + A3
        alpha_Np_m = alpha_dB_km * 0.000115129  # Convert dB/km to Np/m
        
        return alpha_Np_m
    
    def calculate_viscosity(self) -> float:
        """Calculate dynamic viscosity of fluid"""
        if self.name.lower() == "water":
            return self._calculate_water_viscosity()
        else:
            return self.viscosity
    
    def _calculate_water_viscosity(self) -> float:
        """Calculate water viscosity as function of temperature"""
        T = self.temperature
        
        # Vogel-Fulcher-Tammann equation for water viscosity
        A = -3.7188
        B = 578.919
        C = -137.546
        
        log_viscosity = A + B / (T + C)
        viscosity = 10**log_viscosity  # in Pa·s
        
        return viscosity
    
    def get_absorption_at_frequency(self, frequency: float) -> float:
        """Get absorption coefficient at specific frequency"""
        return self.frequency_response.get_absorption_at_frequency(frequency)
    
    def get_impedance_at_frequency(self, frequency: float) -> complex:
        """Get characteristic impedance at specific frequency"""
        sound_speed = self.calculate_sound_speed()
        density = self.calculate_density()
        
        # For fluids, impedance is primarily real
        impedance_real = density * sound_speed
        
        # Small imaginary component due to viscosity
        absorption = self.get_absorption_at_frequency(frequency)
        wavelength = sound_speed / frequency
        impedance_imag = absorption * wavelength * impedance_real / (2 * np.pi)
        
        return complex(impedance_real, impedance_imag)
    
    def to_soxel_properties(self, frequency: float = 1000.0) -> Dict[str, Any]:
        """Convert to Soxel-compatible properties dictionary"""
        return {
            'sound_speed': self.calculate_sound_speed(),
            'density': self.calculate_density(),
            'absorption_coeffs': {frequency: self.get_absorption_at_frequency(frequency)},
            'viscosity': self.calculate_viscosity(),
            'impedance': self.get_impedance_at_frequency(frequency),
            'shader_type': 'fluid',
            'name': self.name
        }
    
    @classmethod
    def from_preset(cls, preset_name: str) -> 'FluidShader':
        """Create fluid shader from common presets"""
        presets = {
            'water_fresh': {
                'name': 'water_fresh',
                'temperature': 20.0,
                'salinity': 0.0
            },
            'water_seawater': {
                'name': 'water_seawater',
                'temperature': 15.0,
                'salinity': 35.0
            },
            'water_hot': {
                'name': 'water_hot',
                'temperature': 50.0,
                'salinity': 0.0
            },
            'oil_engineengine': {
                'name': 'oil_engine',
                'temperature': 20.0,
                'density': 900.0,
                'bulk_modulus': 1.6e9,
                'viscosity': 0.1
            },
            'alcohol': {
                'name': 'alcohol',
                'temperature': 20.0,
                'density': 789.0,
                'bulk_modulus': 0.9e9,
                'viscosity': 0.0012
            },
            'mercury': {
                'name': 'mercury',
                'temperature': 20.0,
                'density': 13534.0,
                'bulk_modulus': 28.5e9,
                'viscosity': 0.0015
            }
        }
        
        if preset_name in presets:
            return cls(**presets[preset_name])
        else:
            raise ValueError(f"Unknown fluid preset: {preset_name}")
    
    def to_dict(self) -> Dict:
        """Convert fluid shader to dictionary"""
        return {
            'name': self.name,
            'temperature': self.temperature,
            'pressure': self.pressure,
            'salinity': self.salinity,
            'viscosity': self.viscosity,
            'bulk_modulus': self.bulk_modulus,
            'density': self.density,
            'frequency_response': self.frequency_response.to_dict() if self.frequency_response else None
        }
    
    def save_to_file(self, file_path: str):
        """Save fluid shader to JSON file"""
        data = self.to_dict()
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load_from_file(cls, file_path: str) -> 'FluidShader':
        """Load fluid shader from JSON file"""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Handle frequency response
        fr_data = data.pop('frequency_response', None)
        shader = cls(**data)
        
        if fr_data and shader.frequency_response:
            shader.frequency_response.load_from_dict(fr_data)
        
        return shader

