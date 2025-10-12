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
Directivity pattern handling for acoustic sources and receivers.
Provides spherical harmonics and pattern interpolation.
"""

import numpy as np
import json
from typing import Dict, List, Tuple, Optional, Callable
from scipy.special import sph_harm
from scipy.interpolate import RegularGridInterpolator
import numba as nb


class DirectivityPattern:
    """Handle 3D directivity patterns with frequency dependence"""
    
    def __init__(self, pattern_data: Optional[Dict] = None):
        self.azimuth = np.array([])  # 0 to 360 degrees
        self.elevation = np.array([])  # -90 to 90 degrees
        self.frequencies = np.array([])  # Frequency bands
        self.pattern = np.array([])  # 3D array: [freq, azimuth, elevation]
        
        if pattern_data:
            self.load_from_dict(pattern_data)
    
    def load_from_dict(self, data: Dict):
        """Load directivity pattern from dictionary"""
        if 'azimuth' in data and 'elevation' in data and 'pattern' in data:
            self.azimuth = np.array(data['azimuth'], dtype=np.float32)
            self.elevation = np.array(data['elevation'], dtype=np.float32)
            self.pattern = np.array(data['pattern'], dtype=np.float32)
            
            if 'frequencies' in data:
                self.frequencies = np.array(data['frequencies'], dtype=np.float32)
            else:
                # Assume single frequency if not specified
                self.frequencies = np.array([1000.0])
        
        elif 'file_path' in data:
            self.load_from_file(data['file_path'])
    
    def load_from_file(self, file_path: str):
        """Load directivity pattern from JSON file"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            self.load_from_dict(data)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading directivity pattern from {file_path}: {e}")
            # Create omnidirectional pattern as fallback
            self._create_omnidirectional_pattern()
    
    def _create_omnidirectional_pattern(self):
        """Create default omnidirectional pattern"""
        self.azimuth = np.linspace(0, 360, 37)  # 10-degree steps
        self.elevation = np.linspace(-90, 90, 19)  # 10-degree steps
        self.frequencies = np.array([1000.0])
        
        # Create uniform pattern (omnidirectional)
        self.pattern = np.ones((1, len(self.azimuth), len(self.elevation)), 
                              dtype=np.float32)
    
    def get_directivity(self, azimuth: float, elevation: float, 
                                             frequency: Optional[float] = None) -> float:
        """
        Get directivity coefficient for given direction and frequency.
        
        Args:
            azimuth: Azimuth angle in degrees (0-360)
            elevation: Elevation angle in degrees (-90 to 90)
            frequency: Frequency in Hz (optional)
        
        Returns:
            Directivity coefficient (linear scale)
        """
        if len(self.pattern) == 0:
            return 1.0  # Omnidirectional fallback
        
        # Normalize angles
        azimuth = azimuth % 360
        elevation = np.clip(elevation, -90, 90)
        
        # Find frequency index
        if frequency is None or len(self.frequencies) == 1:
            freq_idx = 0
        else:
            # Find closest frequency
            freq_idx = np.argmin(np.abs(self.frequencies - frequency))
        
        # Create interpolator for this frequency
        pattern_2d = self.pattern[freq_idx]
        interpolator = RegularGridInterpolator(
            (self.azimuth, self.elevation), 
            pattern_2d,
            method='linear',
            bounds_error=False,
            fill_value=1.0  # Default to omnidirectional outside bounds
        )
        
        # Interpolate
        direction = np.array([[azimuth, elevation]])
        directivity = interpolator(direction)[0]
        
        return float(directivity)
    
    def apply_directivity(self, audio_signal: np.ndarray, 
                         azimuth: float, elevation: float,
                         frequency: Optional[float] = None) -> np.ndarray:
        """
        Apply directivity pattern to audio signal.
        
        Args:
            audio_signal: Input audio signal
            azimuth: Source/receiver azimuth
            elevation: Source/receiver elevation
            frequency: Center frequency (optional)
        
        Returns:
            Directivity-adjusted signal
        """
        directivity_coeff = self.get_directivity(azimuth, elevation, frequency)
        return audio_signal * directivity_coeff
    
    def to_spherical_harmonics(self, order: int = 3, 
                             frequency: Optional[float] = None) -> np.ndarray:
        """
        Convert directivity pattern to spherical harmonics coefficients.
        
        Args:
            order: Maximum spherical harmonics order
            frequency: Target frequency (optional)
        
        Returns:
            Spherical harmonics coefficients
        """
        if len(self.pattern) == 0:
            return np.array([1.0] + [0.0] * ((order + 1) ** 2 - 1))
        
        # Get pattern for target frequency
        if frequency is None:
            pattern_2d = self.pattern[0]  # Use first frequency
        else:
            freq_idx = np.argmin(np.abs(self.frequencies - frequency))
            pattern_2d = self.pattern[freq_idx]
        
        # Convert to spherical coordinates
        n_channels = (order + 1) ** 2
        coefficients = np.zeros(n_channels, dtype=np.complex128)
        
        # Convert degrees to radians for spherical harmonics
        azimuth_rad = np.deg2rad(self.azimuth)
        elevation_rad = np.deg2rad(90 - self.elevation)  # Convert to inclination
        
        # Create meshgrid for spherical coordinates
        AZ, EL = np.meshgrid(azimuth_rad, elevation_rad, indexing='ij')
        
        # Calculate spherical harmonics coefficients
        idx = 0
        for l in range(order + 1):
            for m in range(-l, l + 1):
                # Calculate spherical harmonic
                Y_lm = sph_harm(m, l, AZ, EL)
                
                # Project pattern onto spherical harmonic
                coefficient = np.sum(pattern_2d * Y_lm.conj() * np.sin(EL))
                coefficient /= np.sum(np.sin(EL))  # Normalize
                
                coefficients[idx] = coefficient
                idx += 1
        
        return coefficients
    
    def from_spherical_harmonics(self, coefficients: np.ndarray, 
                               azimuth_res: int = 37, elevation_res: int = 19):
        """
        Reconstruct directivity pattern from spherical harmonics coefficients.
        
        Args:
            coefficients: Spherical harmonics coefficients
            azimuth_res: Number of azimuth points
            elevation_res: Number of elevation points
        """
        # Create angular grid
        self.azimuth = np.linspace(0, 360, azimuth_res)
        self.elevation = np.linspace(-90, 90, elevation_res)
        self.frequencies = np.array([1000.0])
        
        # Convert to spherical coordinates
        azimuth_rad = np.deg2rad(self.azimuth)
        elevation_rad = np.deg2rad(90 - self.elevation)  # Convert to inclination
        
        AZ, EL = np.meshgrid(azimuth_rad, elevation_rad, indexing='ij')
        
        # Reconstruct pattern
        pattern_2d = np.zeros_like(AZ, dtype=np.float32)
        order = int(np.sqrt(len(coefficients)) - 1)
        
        idx = 0
        for l in range(order + 1):
            for m in range(-l, l + 1):
                Y_lm = sph_harm(m, l, AZ, EL)
                pattern_2d += (coefficients[idx] * Y_lm).real
                idx += 1
        
        self.pattern = pattern_2d[np.newaxis, :, :]
    
    def to_dict(self) -> Dict:
        """Convert directivity pattern to dictionary"""
        return {
            'azimuth': self.azimuth.tolist(),
            'elevation': self.elevation.tolist(),
(),
            'frequencies': self.frequencies.tolist(),
            'pattern': self.pattern.tolist()
        }
    
    def save_to_file(self, file_path: str):
        """Save directivity pattern to JSON file"""
        data = self.to_dict()
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)


class CommonDirectivityPatterns:
    """Factory for common directivity patterns"""
    
    @staticmethod
    def omnidirectional() -> DirectivityPattern:
        """Create omnidirectional pattern"""
        pattern = DirectivityPattern()
        pattern._create_omnidirectional_pattern()
        return pattern
    
    @staticmethod
    def cardioid() -> DirectivityPattern:
        """Create cardioid pattern"""
        pattern = DirectivityPattern()
        pattern.azimuth = np.linspace(0, 360, 37)
        pattern.elevation = np.linspace(-90, 90, 19)
        pattern.frequencies = np.array([1000.0])
        
        # Cardioid pattern: 0.5 * (1 + cos(θ))
        AZ, EL = np.meshgrid(pattern.azimuth, pattern.elevation, indexing='ij')
        theta = np.deg2rad(AZ)
        pattern_2d = 0.5 * (1 + np.cos(theta))
        
        pattern.pattern = pattern_2d[np.newaxis, :, :]
        return pattern
    
    @staticmethod
    def figure8() -> DirectivityPattern:
        """Create figure-8 pattern"""
        pattern = DirectivityPattern()
        pattern.azimuth = np.linspace(0, 360, 37)
        pattern.elevation = np.linspace(-90, 90, 19)
        pattern.frequencies = np.array([1000.0])
        
        # Figure-8 pattern: cos(θ)
        AZ, EL = np.meshgrid(pattern.azimuth, pattern.elevation, indexing='ij')
        theta = np.deg2rad(AZ)
        pattern_2d = np.cos(theta)
        
        pattern.pattern = pattern_2d[np.newaxis, :, :]
        return pattern
    
    @staticmethod
    def hypercardioid() -> DirectivityPattern:
        """Create hypercardioid pattern"""
        pattern = DirectivityPattern()
        pattern.azimuth = np.linspace(0, 360, 37)
        pattern.elevation = np.linspace(-90, 90, 19)
        pattern.frequencies = np.array([1000.0])
        
        # Hypercardioid pattern: 0.25 * (1 + 3*cos(θ))
        AZ, EL = np.meshgrid(pattern.azimuth, pattern.elevation, indexing='ij')
        theta = np.deg2rad(AZ)
        pattern_2d = 0.25 * (1 + 3 * np.cos(theta))
        
        pattern.pattern = pattern_2d[np.newaxis, :, :]
        return pattern


@nb.jit(nopython=True)
def apply_directivity_gain(signal: np.ndarray, azimuth: float, elevation: float,
                          directivity_coeff: float) -> np.ndarray:
    """
    Apply directivity gain to signal using numba acceleration.
    
    Args:
        signal: Input audio signal
        azimuth: Azimuth angle in degrees
        elevation: Elevation angle in degrees
        directivity_coeff: Directivity coefficient (linear scale)
    
    Returns:
        Directivity-adjusted signal
    """
    return signal * directivity_coeff


@nb.jit(nopython=True)
def spherical_to_cartesian(azimuth: float, elevation: float, 
                          radius: float = 1.0) -> Tuple[float, float, float]:
    """
    Convert spherical coordinates to cartesian coordinates.
    
    Args:
        azimuth: Azimuth angle in degrees
        elevation: Elevation angle in degrees
        radius: Radial distance
    
    Returns:
        (x, y, z) coordinates
    """
    az_rad = np.deg2rad(azimuth)
    el_rad = np.deg2rad(elevation)
    
    x = radius * np.cos(el_rad) * np.cos(az_rad)
    y = radius * np.cos(el_rad) * np.sin(az_rad)
    z = radius * np.sin(el_rad)
    
    return x, y, z


@nb.jit(nopython=True)
def cartesian_to_spherical(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """
    Convert cartesian coordinates to spherical coordinates.
    
    Args:
        x, y, z: Cartesian coordinates
    
    Returns:
        (azimuth, elevation, radius) in degrees and units
    """
    radius = np.sqrt(x*x + y*y + z*z)
    
    if radius == 0:
        return 0.0, 0.0, 0.0
    
    azimuth = np.rad2deg(np.arctan2(y, x)) % 360
    elevation = np.rad2deg(np.arcsin(z / radius))
    
    return azimuth, elevation, radius

