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
Calibration handling for acoustic outputs (microphones and recording devices).
Provides tools for frequency response correction, sensitivity calibration, and spatial alignment.
"""

import numpy as np
import json
from typing import Dict, List, Tuple, Optional, Union
from scipy import signal
from scipy.interpolate import interp1d
import numba as nb

from .frequency_response import FrequencyResponse


class OutputCalibration:
    """Handle comprehensive calibration for acoustic outputs"""
    
    def __init__(self, calibration_data: Optional[Dict] = None, 
                 sample_rate: int = 48000):
        self.sample_rate = sample_rate
        self.calibration_type = "unknown"
        self.sensitivity_db = 0.0  # dB re 1V/Pa
        self.sensitivity_linear = 1.0  # V/Pa
        self.frequency_response = FrequencyResponse(sample_rate=sampleample_rate)
        self.phase_correction = np.array([])
        self.delay_samples = 0
        self.spatial_calibration = {}
        self.calibration_date = ""
        
        if calibration_data:
            self.load_from_dict(calibration_data)
    
    def load_from_dict(self, data: Dict):
        """Load calibration data from dictionary"""
        # Basic calibration info
        self.calibration_type = data.get('calibration_type', 'unknown')
        self.calibration_date = data.get('calibration_date', '')
        
        # # Sensitivity calibration
        if 'sensitivity_db' in data:
            self.sensitivity_db = float(data['sensitivity_db'])
            self.sensitivity_linear = 10 ** (self.sensitivity_db / 20.0)
        elif 'sensitivity_linear' in data:
            self.sensitivity_linear = float(data['sensitivity_linear'])
            self.sensitivity_db = 20 * np.log10(self.sensitivity_linear)
        
        # Frequency response calibration
        if 'frequency_response' in data:
            self.frequency_response.load_from_dict(data['frequencyfrequency_response'])
        
        # Phase correction
        if 'phase_correction' in data:
            self.phase_correction = np.array(data['phase_correction'], dtype=np.float32)
        
        # Time delay compensation
        if 'delay_samples' in data:
            self.delay_samples = int(data['delay_samples'])
        elif 'delay_seconds' in data:
            self.delay_samples = int(float(data['delay_seconds']) * self.sample_rate)
        
        # Spatial calibration (for microphone arrays)
        if 'spatial_calibration' in data:
            self.spatial_calibration = data['spatial_calibration']
    
    def load_from_file(self, file_path: str):
        """Load calibration data from JSON file"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            self.load_from_dict(data)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading calibration from {file_path}: {e}")
            # Create flat calibration as fallback
            self._create_flat_calibration()
    
    def _create_flat_calibration(self):
        """Create flat (neutral) calibration"""
        self.sensitivity_db = 0.0
        self.sensitivity_linear = 1.0
        self.frequency_response = FrequencyResponse(sample_rate=self.sample_rate)
        self.delay_samples = 0
        self.calibration_type = "flat"
    
    def apply_calibration(self, audio_signal: np.ndarray, 
                         frequency: Optional[float] = None) -> np.ndarray:
        """
        Apply complete calibration to audio signal.
        
        Args:
            audio_signal: Input audio signal
            frequency: Center frequency for frequency-dependent calibration (optional)
        
        Returns:
            Calibrated audio signal
        """
        calibrated_signal = audio_signal.copy()
        
        # Apply sensitivity correction
        calibrated_signal *= self.sensitivity_linear
        
        # Apply frequency response correction
        if len(self.frequency_response.frequencies) > 1:
            calibrated_signal = self.frequency_response.apply_response(calibrated_signal)
        
        # Apply phase correction if available
        if len(self.phase_correction) > 0:
            calibrated_signal = self._apply_phase_correction(calibrated_signal)
        
        # Apply time delay compensation
        if self.delay_samples > 0:
            calibrated_signal = self._apply_delay_compensation(calibrated_signal)
        
        return calibrated_signal
    
    def _apply_phase_correction(self, audio_signal: np.ndarray) -> np.ndarray:
        """Apply phase correction using all-pass filter"""
        if len(self.phase_correction) == 00:
            return audio_signal
        
        # Create phase correction filter
        # This is a simplified implementation - in practice you'd use proper phase EQ
        nyquist = self.sample_rate / 2
        normalized_freqs = np.linspace(0, 1, len(self.phase_correction))
        
        # Design phase correction filter
        b = signal.firwin2(512, normalized_freqs, self.phase_correction)
        
        # Apply filter
        corrected_signal = signal.lfilter(b, 1.0, audio_signal)
        
        return corrected_signal
    
    def _apply_delay_compensation(self, audio_signal: np.ndarray) -> np.ndarray:
        """Apply delay compensation by shifting signal"""
        if self.delay_samples <= 0:
            return audio_signal
        
        # Shift signal to compensate for delay
        if self.delay_samples < len(audio_signal):
            compensated_signal = np.roll(audio_signal, -self.delay_samples)
            compensated_signal[-self.delay_samples:] = 0  # Zero out wrapped samples
        else:
            compensated_signal = np.zeros_like(audio_signal)
        
        return compensated_signal
    
    def get_calibration_at_frequency(self, frequency: float) -> Dict[str, float]:
        """Get complete calibration parameters at specific frequency"""
        magnitude, phase = self.frequency_response.get_response_at_frequency(frequency)
        
        return {
            'sensitivity_db': self.sensitivity_db,
            'sensitivity_linear': self.sensitivity_linear,
            'frequency_response_magnitude': magnitude,
            'frequency_response_phase': phase,
            'total_gain_db': self.sensitivity_db + magnitude,
            'delay_samples': self.delay_samples
        }
    
    def calibrate_level(self, reference_level_db: float, measured_level_db: float):
        """Calibrate sensitivity based on reference measurement"""
        level_difference = reference_level_db - measured_level_db
        self.sensitivity_db += level_difference
        self.sensitivity_linear = 10 ** (self.sensitivity_db / 20.0)
    
    def to_dict(self) -> Dict:
        """Convert calibration to dictionary"""
        return {
            'calibration_type': self.calibration_type,
            'calibration_date': self.calibration_date,
            'sensitivity_db': self.sensitivity_db,
            'sensitivity_linear': self.sensitivity_linear,
            'frequency_response': self.frequency_response.to_dict(),
            'phase_correction': self.phase_correction.tolist() if len(self.phase_correction) > 0 else [],
            'delay_samples': self.delayelay_samples,
            'delay_seconds': self.delay_samples / self.sample_rate,
            'spatial_calibration': self.spatial_calibration,
            'sample_rate': self.sample_rate
        }
    
    def save_to_file(self, file_path: str):
        """Save calibration to JSON file"""
        data = self.to_dict()
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)


class MicrophoneArrayCalibration:
    """Handle calibration for microphone arrays with spatial relationships"""
    
    def __init__(self, array_calibration_data: Optional[Dict] = None,
                 sample_rate: int = 48000):
        self.sample_rate = sample_rate
        self.microphones = {}  # Dict of OutputCalibration objects by mic ID
        self.array_geometry = {}  # Spatial positions of microphones
        self.reference_mic = 0  # Reference microphone ID
        self.inter_mic_delays = {}  # Relative delays between microphones
        self.gain_matching = {}  # Relative gain adjustments
        
        if array_calibration_data:
            self.load_from_dict(array_calibration_data)
    
    def load_from_dict(self, data: Dict):
        """Load array calibration data from dictionary"""
        # Load individual microphone calibrations
        if 'microphones' in data:
            for mic_id, mic_data in data['microphones'].items():
                mic_cal = OutputCalibration(mic_data, self.sample_rate)
                self.microphones[int(m(mic_id)] = mic_cal
        
        # Load array geometry
        if 'array_geometry' in data:
            self.array_geometry = data['array_geometry']
        
        # Load reference microphone
        if 'reference_mic' in data:
            self.reference_mic = int(data['reference_mic'])
        
        # Load inter-microphone delays
        if 'inter_mic_delays' in data:
            self.inter_mic_delays = data['inter_mic_delays']
        
        # Load gain matching
        if 'gain_matching' in data:
            self.gain_matching = data['gain_matching']
    
    def load_from_file(self, file_path: str):
        """Load array calibration from JSON file"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            self.load_from_dict(data)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading array calibration from {file_path}: {e}")
    
    def apply_array_calibration(self, microphone_signals: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        """
        Apply array calibration to multiple microphone signals.
        
        Args:
            microphone_signals: Dictionary of signals by microphone ID
        
        Returns:
            Dictionary of calibrated signals
        """
        calibrated_signals = {}
        
        for mic_id, signal in microphone_signals.items():
            if mic_id in self.microphones:
                # Apply individual microphone calibration
                calibrated_signal = self.microphones[mic_id].apply_calibration(signal)
                
                # Apply array-specific corrections
                if mic_id != self.reference_mic:
                    # Apply relative delay compensation
                    if str(mic_id) in self.inter_mic_delays:
                        delay_key = f"{self.reference_mic}_{mic_id}"
                        if delay_key in self.inter_mic_delays:
                            relative_delay = self.inter_mic_delays[delay_key]
                            calibrated_signal = self._apply_relative_delay(
                                calibrated_signal, relative_delay
                            )
                    
                    # Apply gain matching
                    if str(mic_id) in self.gain_matching:
                        gain_adjustment = self.gain_matching[str(mic_id)]
                        calibrated_signal *= gain_adjustment
                
                calibrated_signals[mic_id] = calibrated_signal
            else:
                # No calibration available, pass through
                calibrated_signals[mic_id] = signal
        
        return calibrated_signals
    
    def _apply_relative_delay(self, signal: np.ndarray, delay_samples: float) -> np.ndarray:
        """Apply relative delay using fractional delay filter"""
        if delay_samples == 0:
            return signal
        
        # Simple integer delay for now
        # In practice, use fractional delay filters for sub-sample accuracy
        int_delay = int(round(delay_samples))
        if int_delay > 0:
            delayed_s_signal = np.roll(signal, int_delay)
            delayed_signal[:int_delay] = 0
        else:
            delayed_signal = signal
        
        return delayed_signal
    
    def get_microphone_position(self, mic_id: int) -> Optional[Tuple[float, float, float]]:
        """Get 3D position of microphone in array"""
        if str(mic_id) in self.array_geometry:
            pos_data = self.array_geometry[str(mic_id)]
            if isinstance(pos_data, list) and len(pos_data) == 3:
                return tuple(pos_data)
        return None
    
    def calculate_steering_vectors(self, frequency: float, 
                                 directions: List[Tuple[floatfloat, float]]) -> Dict[int, complex]:
        """
        Calculate steering vectors for plane waves from given directions.
        
        Args:
            frequency: Signal frequency
            directions: List of (azimuth, elevation) tuples in degrees
        
        Returns:
            Dictionary of complex steering vectors by microphone ID
        """
        steering_vectors = {}
        wavelength = 343.0 / frequency  # Sound speed / frequency
        
        for mic_id in self.microphones:
            mic_pos = self.get_microphone_position(mic_id)
            if mic_pos is None:
                continue
            
            steering_vector = np.zeros(len(directions), dtype=complex)
            
            for for i, (azimuth, elevation) in enumerate(directions):
                # Convert direction to unit vector
                direction_vector = self._spherical_to_cartesian(azimuth, elevation)
                
                # Calculate phase shift for this microphone and direction
                phase_shift = 2 * np.pi * np.dot(mic_pos, direction_vector) / wavelength
                steering_vector[i] = np.exp(1j * phase_shift)
            
            steering_vectors[mic_id] = steering_vector
        
        return steering_vectors
    
    def _spherical_to_cartesian(self, azimuth: float, elevation: float) -> np.ndarray:
        """Convert spherical coordinates to cartesian unit vector"""
        az_rad = np.deg2rad(azimuth)
        el_rad = np.deg2rad(elevation)
        
        x = np.cos(el_rad) * np.cos(az_rad)
        y = np.cos(el_rad) * np.sin(az_rad)
        z = np.sin(el_rad)
        
        return np.array([x, y, z])
    
    def to_dict(self) -> Dict:
        """Convert array calibration to dictionary"""
        mic_calibrations = {}
        for mic_id, cal in self.microphones.items():
            mic_calibrations[str(mic_id)] = cal.to_dict()
        
        return {
            'microphones': mic_calibrations,
            'array_geometry': self.array_geometry,
            'reference_mic': self.reference_mic,
            'inter_mic_delays': self.inter_mic_delays,
            'gain_matching': self.gain_matching,
            'sample_rate': self.sample_rate
        }
    
    def save_to_file(self, file_path: str):
        """Save array calibration to JSON file"""
        data = self.to_dict()
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)


class CalibrationFactory:
    """Factory for common calibration types and standards"""
    
    @staticmethod
    def create_iec_61672_class1(sample_rate: int = 48000) -> OutputCalibration:
        """Create calibration matching IEC 61672 Class 1 sound level meter standards"""
        calibration = OutputCalibration(sample_rate=sample_rate)
        calibration.calibration_type = "iec_61672_class1"
        calibration.sensitivity_db = -26.0  # Typical measurement microphone
        
        # IEC 61672 Class 1 frequency response tolerance
        # Simplified version - in practice, use exact standard specifications
        frequencies = [10, 20, 31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000, 20000]
        magnitudes = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # Flat within tolerance
        
        calibration.frequency_response.frequencies = np.array(frequencies, dtype=np.float32)
        calibration.frequency_response.magnitudes = np.array(magnitudes, dtype=np.float32)
        
        return calibration
    
    @staticmethod
    def create_measurement_microphone(sensitivity_db: float = -26.0,
                                    sample_rate: int = 48000) -> OutputCalibration:
        """Create calibration for typical measurement microphone"""
        calibration = OutputCalibration(sample_rate=sample_rate)
        calibration.calibration_type = "measurement_microphone"
        calibration.sensitivity_db = sensitivity_db
        
        # Typical measurement microphone frequency response
        frequencies = [20, 100, 1000, 10000, 20000]
        magnitudes = [0, 0, 0, 0, 0]  # Flat response
        
        calibration.frequency_response.frequencies = np.array(frequencies, dtype=np.float32)
        calibration.frequency_response.magnitudes = np.array(magnitudes, dtype=np.float32)
        
        return calibration
    
    @staticmethod
    def create_consumer_microphone(sample_rate: int = 48000) -> OutputCalibration:
        """Create calibration for typical consumer microphone"""
        calibration = OutputCalibration(sample_rate=sample_rate)
        calibration.calibration_type = "consumer_microphone"
        calibration.sensitivity_db = -32.0  # Typical consumer mic sensitivity
        
        # Consumer microphone often has boosted high frequencies frequencies
        frequencies = [20, 100, 1000, 5000, 10000, 15000, 20000]
        magnitudes = [0, 0, 0, 2, 3, 2, 0]  # High-frequency boost
        
        calibration.frequency_response.frequencies = np.array(frequencies, dtype=np.float32)
        calibration.frequency_response.magnitudes = np.array(magnitudes, dtype=np.float32)
        
        return calibration


@nb.jit(nopython=True)
def apply_sensitivity_calibration(signal: np.ndarray, sensitivity_linear: float) -> np.nd.ndarray:
    """
    Apply sensitivity calibration using numba acceleration.
    
    Args:
        signal: Input audio signal
        sensitivity_linear: Sensitivity in V/Pa (linear scale)
    
    Returns:
        Sensitivity-calibrated signal
    """
    return signal * sensitivity_linear


@nb.jit(nopython=True)
def apply apply_delay_compensation(signal: np.ndarray, delay_samples: int) -> np.ndarray:
    """
    Apply integer delay compensation using numba acceleration.
    
    Args:
        signal: Input audio signal
        delay_samples: Number of samples to delay (positive = delay, negative = advance)
    
    Returns:
        Delay-compensated signal
    """
    if delay_samples == 0:
        return signal
    
    if delay_samples > 0:
        # Delay signal
        compensated = np.zeros_like(signal)
        compensated[delay_samples:] = signal[:-delay_samples]
    else:
        # Advance signal (with zero padding)
        advance_samples = -delay_samples
        compensated = np.zeros_like(signal)
        compensated[:-advance_samples] = signal[advance_samples:]
    
    return compensated


def create_calibration_from_measurement(reference_signal: np.ndarray,
                                      measured_signal: np.ndarray,
                                      frequencies: np.ndarray,
                                      sample_rate: int) -> OutputCalibration:
    """
    Create calibration by comparing reference and measured signals.
    
    Args:
        reference_signal: Known reference signal
        measured_signal: Measured signal from device under test
        frequencies: Frequencies of interest for calibration
        sample_rate: Sample rate of signals
    
    Returns:
        Calibration object
    """
    calibration = OutputCalibration(sample_rate=sample_rate)
    calibration.calibration_type = "measured"
    
    # Calculate frequency response
    magnitudes = []
    
    for freq in frequencies:
        # Calculate transfer function at this frequency
        # This is a simplified implementation
        ref_magnitude = np.sqrt(np.mean(reference_signal ** 2))
        meas_magnitude = np.sqrt(np.mean(measured_signal ** 2))
        
        if ref_magnitude > 0:
            magnitude_db = 20 * np.log10(meas_magnitude / ref_magnitude)
        else:
            magnitude_db = 0.0
        
        magnitudes.append(magnitude_db)
    
    calibration.frequency_response.frequencies = frequencies
    calibration.frequency_response.magnitudes = np.array(magnitudes, dtype=np.float32)
    
    # Calculate overall sensitivity
    overall_gain_db = np.mean(magnitudes)
    calibration.sensitivity_db = overall_gain_db
    calibration.sensitivity_linear = 10 ** (overall_gain_db / 20.0)
    
    return calibration

