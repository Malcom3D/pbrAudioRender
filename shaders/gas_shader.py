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
Gas medium shader with frequency-dependent acoustic properties.
Implements atmospheric and gaseous media with temperature, pressure, and humidity effects.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json

from ..lib.frequency_response import MaterialFrequencyResponse


@dataclass
class GasShader:
    """Acoustic shader for gaseous media (air, helium, CO2, etc.)"""
    
    name: str = "air"
    temperature: float = 20.0  # Celsius
    pressure: float = 101325.0  # Pascals
    humidity: float = 50.0  # Relative humidity %
    gas_composition: Dict[str, float] = None
    frequency_response: MaterialFrequencyResponse = None
    
    def __post_init__(self):
        if self.gas_composition is None:
            self.gas_composition = {'N2': 78.08, 'O2': 20.95, 'Ar': 0.93, 'CO2': 0.04}
        
        if self.frequency_response is None:
            self.frequency_response = MaterialFrequencyResponse()
            self._initialize_default_response()
    
    def _initialize_default_response(self):
        """Initialize default frequency response for air"""
        # Default atmospheric absorption coefficients (ISO 9613-1)
        frequencies = np.array([63, 125, 250, 500, 1000, 2000, 4000, 8000], dtype=float)
        absorption = self.calculate_atmospheric_absorption(frequencies)
        
        self.frequency_response.frequencies = frequencies
        self.frequency_response.absorption_coeffs = absorption
    
    def calculate_sound_speed(self) -> float:
        """
        Calculate sound speed in gas using ideal gas law with humidity correction.
        
        Returns:
            Sound speed in m/s
        """
        T = self.temperature + 273.15  # Convert to Kelvin
        
        # Calculate molar mass of gas mixture
        molar_masses = {
            'N2': 28.0134, 'O2': 31.9988, 'Ar': 39.948, 'CO2': 44.0095,
            'H2O': 18.01528
        }
        
        # Calculate effective molar mass considering humidity
        water_vapor_pressure = self._calculate_water_vapor_pressure()
        dry_air_pressure = self.pressure - water_vapor_pressure
        
        # Weighted average molar mass
        total_molar_mass = 0.0
        for gas, fraction in self.gas_composition.items():
            if gas in molar_masses:
                partial_pressure = dry_air_pressure * (fraction / 100.0)
                total_molar_mass += molar_masses[gas] * (partial_pressure / self.pressure)
        
        # Add water vapor contribution
        total_molar_mass += molar_masses['H2O'] * (water_vapor_pressure / self.pressure)
        
        # Gas constant
        R = 8.314462618  # J/(mol·K)
        gamma = self._calculate_adiabatic_index()
        
        # Sound speed formula: c = √(γ * R * T / M)
        sound_speed = np.sqrt(gamma * R * T / (total_molar_mass / 1000))  # Convert g/mol to kg/mol
        
        return float(sound_speed)
    
    def _calculate_water_vapor_pressure(self) -> float:
        """Calculate water vapor pressure from humidity and temperature"""
        # Magnus formula for saturation vapor pressure
        T = self.temperature
        saturation_pressure = 611.21 * np.exp((18.678 - T/234.5) * (T / (257.14 + T)))
        
        # Actual vapor pressure
        vapor_pressure = saturation_pressure * (self.humidity / 100.0)
        
        return vapor_pressure
    
    def _calculate_adiabatic_index(self) -> -> float:
        """Calculate adiabatic index (ratio of specific heats) for gas mixture"""
        # Specific heat ratios for common gases
        gamma_values = {
            'N2': 1.4, 'O2': 1.4, 'Ar': 1.67, 'CO2': 1.28, 'H2O': 1.33
        }
        
        # Weighted average based on composition
        total_gamma = 0.0
        total_fraction = 0.0
        
        for gas, fraction in self.gas_composition.items():
            if gas in gamma_values:
                total_gamma += gamma_values[gas] * fraction
                total_fraction += fraction
        
        if total_fraction > 0:
            return total_gamma / total_fraction
        else:
            return 1.4  # Default for air
    
    def calculate_density(self) -> float:
        """
        Calculate gas density using ideal gas law.
        
        Returns:
            Density in kg/m³
        """
        T = self.temperature + 273.15  # Kelvin
        R_specific = 287.05  # Specific gas constant for dry air (J/(kg·K))
        
        # Adjust for humidity
        water_vapor_pressure = self._calculate_water_vapor_pressure()
        dry_air_pressure = self.pressure - water_vapor_pressure
        
        # Density of dry air component
        rho_dry = dry_air_pressure / (R_specific * T)
        
        # Density of water vapor component
        R_water = 461.5  # J/(kg·K)
        rho_vapor = water_vapor_pressure / (R_water * T)
        
        total_density = rho_dry + rho_vapor
        
        return float(total_density)
    
    def calculate_atmospheric_absorption(self, frequencies: np.ndarray) -> np.ndarray:
        """
        Calculate atmospheric absorption coefficients (ISO 9613-1).
        
        Args:
            frequencies: Array of frequencies in Hz
        
        Returns:
            Array of absorption coefficients in Np/m
        """
        T = self.temperature + 273.15  # Kelvin
        P0 = 101325.0  # Reference pressure (Pa)
        T0 = 293.15    # Reference temperature (K)
        
        # Calculate relaxation frequencies for oxygen and nitrogen
        frO = (P0 / self.pressure) * (24 + 4.04e4 * self.humidity * (0.02 + self.humidity) / 
                                     (0.391 + self.humidity))
        frN = (P0 / self.pressure) * np.sqrt(T0 / T) * (
            9 + 280 * self.humidity * np.exp(-4.17 * ((T0 / T)**(1/3) - 1))
        )
        
        # Calculate absorption coefficient
        alpha = frequencies**2 * (
            1.84e-11 * (P0 / self.pressure) * np.sqrt(T / T0) +
            (T / T0)**(-2.5) * (
                0.01275 * np.exp(-2239.1 / T) / (frO + frequencies**2 / frO) +
                0.1068 * np.exp(-3352.0 / T) / (frN + frequencies**2 / frN)
            )
        )
        
        return alpha
    
    def get_absorption_at_frequency(self, frequency: float) -> float:
        """Get absorption coefficient at specific frequency"""
        return self.frequency_response.get_absorption_at_frequency(frequency)
    
    def get_impedance_at_frequency(self, frequency: float) -> complex:
        """Get characteristic impedance at specific frequency"""
        sound_speed = self.calculate_sound_speed()
        density = self.calculate_density()
        
        # For gases, impedance is primarily real (resistive)
        impedance_real = density * sound_speed
        
        # Small imaginary component due to absorption
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
            'impedance': self.get_impedance_at_frequency(frequency),
            'shader_type': 'gas',
            'name': self.name
        }
    
    @classmethod
    def from_preset(cls, preset_name: str) -> 'GasShader':
        """Create gas shader from common presets"""
        presets = {
            'air_standard': {
                'name': 'air_standard',
                'temperature': 20.0,
                'pressure': 101325.0,
                'humidity': 50.0
            },
            'air_dry': {
                'name': 'air_dry',
                'temperature': 20.0,
                'pressure': 101325.0,
                'humidity': 0.0
            },
            'helium': {
                'name': 'helium',
                'temperature': 20.0,
                'pressure': 101325.0,
                'gas_composition': {'He': 100.0}
            },
            'carbon_dioxide': {
                'name': 'carbon_dioxide',
                'temperature': 20.0,
                'pressure': 101325.0,
                'gas_composition': {'CO2': 100.0}
            },
            'hot_air': {
                'name': 'hot_air',
                'temperature': 40.0,
                'pressure': 101325.0,
                'humidity': 30.0
            },
            'cold_air': {
                'name': 'cold_air',
                'temperature': 0.0,
                'pressure': 101325.0,
                'humidity': 80.0
            }
        }
        
        if preset_name in presets:
            return cls(**presets[preset_name])
        else:
            raise ValueError(f"Unknown gas preset: {preset_name}")
    
    def to_dict(self) -> Dict:
        """Convert gas shader to dictionary"""
        return {
            'name': self.name,
            'temperature': self.temperature,
            'pressure': self.pressure,
            'humidity': self.humidity,
            'gas_composition': self.gas_composition,
            'frequency_response': self.frequency_response.to_dict() if self.frequency_response else None
        }
    
    def save_to_file(self, file_path: str):
        """Save gas shader to JSON file"""
        data = self.to_dict()
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load_from_file(cls, file_path: str) -> 'GasShader':
        """Load gas shader from JSON file"""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Handle frequency response
        fr_data = data.pop('frequency_response', None)
        shader = cls(**data)
        
        if fr_data and shader.frequency_response:
            shader.frequency_response.load_from_dict(fr_data)
        
        return shader

