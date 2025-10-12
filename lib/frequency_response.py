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
Frequency response handling for acoustic sources and materials.
Provides tools for frequency-dependent gain, phase, and filtering.
"""

import numpy as np
import json
from typing import Dict, List, Tuple, Optional, Union
from scipy import signal
from scipy.interpolate import interp1d
import numba as nb


class FrequencyResponse:
    """Handle frequency-dependent response with interpolation and filtering"""
    
    def __init__(self, frequency_response_data: Optional[Dict] = None, 
                 sample_rate: int = 48000):
        self.sample_rate = sample_rate
        self.frequencies = np.array([])
        self.magnitudes = np.array([])
        self.phases = np.array([])
        
        if frequency_response_data:
            self.load_from_dict(frequency_response_data)
    
    def load_from_dict(self, data: Dict):
        """Load frequency response from dictionary"""
        if 'frequencies' in data and 'magnitudes' in data:
            self.frequencies = np.array(data['frequencies'], dtype=np.float32)
            self.magnitudes = np.array(data['magnitudes'], dtype=np.float32)
            
            if 'phases' in data:
                self.phases = np.array(data['phases'], dtype=np.float32)
            else:
                self.phases = np.zeros_like(self.frequencies)
        
        elif 'file_path' in data:
            self.load_from_file(data['file_path'])
    
    def load_from_file(self, file_path: str):
        """Load frequency response from JSON file"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            self.load_from_dict(data)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading frequency response from {file_path}: {e}")
            # Create flat response as fallback
                       self.frequencies = np.array([20, 20000], dtype=np.float32)
            self.magnitudes = np.array([0, 0], dtype=np.float32)
            self.phases = np.array([0, 0], dtype=np.float32)
    
    def get_response_at_frequency(self, frequency: float) -> Tuple[float, float]:
        """Get magnitude and phase at specific frequency using interpolation"""
        if len(self.frequencies) == 0:
            return 1.0, 0.0  # Flat response
        
        # Handle out-of-bounds frequencies
        if frequency <= self.frequencies[0]:
            return float(self.magnitudes[0]), float(self.phases[0])
        if frequency >= self.frequencies[-1]:
            return float(self.magnitudes[-1]), float(self.phases[-1])
        
        # Interpolate magnitude and phase
        mag_interp = interp1d(self.frequencies, self.magnitudes, 
                             kind='linear', fill_value='extrapolate')
        phase_interp = interp1d(self.frequencies, self.phases,
                               kind='linear', fill_value='extrapolate')
        
        magnitude = float(mag_interp(frequency))
        phase = float(phase_interp(frequency))
        
        return magnitude, phase
    
    def apply_response(self, audio_signal: np.ndarray, 
                      frequencies: Optional[np.ndarray] = None) -> np.ndarray:
        """Apply frequency response to audio signal"""
        if len(self.frequencies) <= 1:
            return audio_signal  # No meaningful response defined
        
        # Create FIR filter from frequency response
        fir_filter = self._create_fir_filter()
        
        # Apply filter
        filtered_signal = signal.lfilter(fir_filter, 1.0, audio_signal)
        
        return filtered_signal
    
    def _create_fir_filter(self, numtaps: int = 512) -> np.ndarray:
        """Create FIR filter from frequency response"""
        # Normalize frequencies to 0-1 (relative to Nyquist)
        nyquist = self.sample_rate / 2
        normalized_freqs = self.frequencies / nyquist
        
        # Ensure frequencies are within [0, 1]
        normalized_freqs = np.clip(normalized_freqs, 0.0, 1.0)
        
        # Convert magnitude to linear scale if in dB
        if np.any(self.magnitudes < 0):
            # Assume magnitudes are in dB, convert to linear
            magnitudes_linear = 10 ** (self.magnitudesitudes / 20.0)
        else:
            magnitudes_linear = self.magnitudes
        
        # Design FIR filter
        fir_coeffs = signal.firwin2(numtaps, normalized_freqs, magnitudes_linear)
        
        return fir_coeffs
    
    def get_octave_band_response(self, center_frequencies: List[float]) -> Dict[float, float]:
        """Get response averaged over octave bands"""
        band_responses = {}
        
        for center_freq in center_frequencies:
            # Define octave band
            low_freq = center_freq / np.sqrt(2)
            high_freq = center_freq * np.sqrt(2)
            
            # Get response within band
            band_indices = np.where((self.frequencies >= low_freq) & 
                                  (self.frequencies <= high_freq))[0]
            
            if len(band_indices) > 0:
                band_magnitude = np.mean(self.magnitudes[band_indices])
                band_responses[center_freq] = band_magnitude
            else:
                # Interpolate if no direct measurements in band
                magnitude, _ = self.get_response_at_frequency(center_freq)
                band_responses[center_freq] = magnitude
        
        return band_responses
    
    def to_dict(self) -> Dict:
        """Convert frequency response to dictionary"""
        return {
            'frequencies': self.frequencies.tolist(),
            'magnitudes': self.magnitudes.tolist(),
            'phases': self.phases.tolist(),
            'sample_rate': self.sample_rate
        }
    
    def save_to_file(self, file_path: str):
        """Save frequency response to JSON file"""
        data = self.to_dict()
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)


class MaterialFrequencyResponse(FrequencyResponse):
    """Extended frequency response for acoustic materials"""
    
    def __init__(self, material_data: Optional[Dict] = None, sample_rate: int = 48000):
        super().__init__(sample_rate=sample_rate)
        
        # Material-specific properties
        self.impedance = np.array([])
        self.absorption_coeffs = np.array([])
        self.scattering_coeffs = np.array([])
        
        if material_data:
            self.load_material_data(material_data)
    
    def load_material_data(self, material_data: Dict):
        """Load material frequency-dependent properties"""
        if 'impedance' in material_data:
            self.impedance = np.array(material_data['impedance'], dtype=np.complex64)
        
        if 'absorption_coeffs' in material_data:
            abs_data = material_data['absorption_coeffs']
            if 'frequencies' in abs_data and 'coefficients' in abs_data:
                self.frequencies = np.array(abs_data['frequencies'], dtype=np.float32)
                self.absorption_coeffs = np.array(abs_data['coefficients'], dtype=np.float32)
        
        if 'scattering_coeffs' in material_data:
            scatter_data = material_data['scattering_coeffs']
            if 'frequencies' in scatter_data and 'coefficients' in scatter_data:
                # Use same frequencies if not already set
                if len(self.frequencies) == 0:
                    self.frequencies = np.array(scatter_data['frequencies'], dtype=np.float32)
                self.scattering_coeffs = np.array(scatter_data['coefficients'], dtype=np.float32)
    
    def get_absorption_at_frequency(self, frequency: float) -> float:
        """Get absorption coefficient at specific frequency"""
        return self._interpolate_property(self.absorption_coeffs, frequency)
    
    def get_scattering_at_frequency(self, frequency: float) -> float:
        """Get scattering coefficient at specific frequency"""
        return self._interpolate_property(self.scattering_coeffs, frequency)
    
    def get_impedance_at_frequency(self, frequency: float) -> complex:
        """Get complex impedance at specific frequency"""
        if len(self.impedance) == 0:
            return complex(1.0, 0.0)  # Default impedance
        
        # Interpolate real and imaginary parts separately
        real_interp = interp1d(self.frequencies, self.impedance.real,
                              kind='linear', fill_value='extrapolate')
        imag_interp = interp1d(self.frequrequencies, self.impedance.imag,
                              kind='linear', fill_value='extrapolate')
        
        return complex(real_interp(frequency), imag_interp(frequency))
    
    def _interpolate_property(self, property_array: np.ndarray, frequency: float) -> float:
        """Interpolate property array at given frequency"""
        if len(property_array) == 0:
            return 0.0
        
        if frequency <= self.frequencies[0]:
            return float(property_array[0])
        if frequency >= self.frequencies[-1]:
            return float(property_array[-1])
        
        interp_func = interp1d(self.frequencies, property_array,
                              kind='linear', fill_value='extrapolate')
        return float(interp_func(frequency))


@nb.jit(nopython=True)
def apply_frequency_gain(signal: np.ndarray, frequency: float, 
                        gain_db: float, sample_rate: int) -> np.ndarray:
    """
    Apply frequency-specific gain to signal using numba acceleration.
    
    Args:
        signal: Input audio signal
        frequency: Target frequency for gain application
        gain_db: Gain in decibels
        sample_rate: Sample rate of the signal
    
    Returns:
        Gain-adjusted signal
    """
    gain_linear = 10.0 ** (gain_db / 20.0)
    
    # Simple gain application - in practice you'd use filtering
    # This is a simplified version for demonstration
    return signal * gain_linear


def create_band_pass_filters(center_frequencies: List[float], 
                           sample_rate: int, 
                           octave_fraction: float = 1.0) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create band-pass filters for frequency bands.
    
    Args:
        center_frequencies: List of center frequencies
        sample_rate: Audio sample rate
        octave_fraction: Fraction of octave for bandwidth (1.0 = full octave)
    
    Returns:
        List of (b, a) filter coefficients
    """
    filters = []
    nyquist = sample_rate / 2
    
    for center_freq in center_frequencies:
        # Calculate band edges
        bandwidth = center_freq * (2 ** (octave_fraction / 2) - 2 ** (-octave_fraction / 2))
        low_cut = max(center_freq - bandwidth / 2, 20.0)
        high_cut = min(center_freq + bandwidth / 2, nyquist * 0.99)
        
        # Normalize frequencies
        low_norm = low_cut / nyquist
        high_norm = high_cut / nyquist
        
        # Design Butterworth band-pass filter
        b, a = signal.butter(4, [low_norm, high_norm], btype='band')
        filters.append((b, a))
    
    return filters

