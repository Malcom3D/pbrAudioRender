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

from dataclasses import dataclass, field
from typing import Tuple, List, Dict

@dataclass
class VelocityVectors:
    """Contains velocity vectors data"""
    x: float = 0.000001
    y: float = 0.000001
    z: float = 0.000001

@dataclass
class FrequencyLimitedField:
    """Contains frequency band-limited pressure and velocity vector of acoustic field"""
    low_freq: float = 0.000001
    high_freq: float = 0.000001
    pressure: float = 0.000001
    velocity: VelocityVectors = field(default_factory=lambda: VelocityVectors(0.0, 0.0, 0.0))

@dataclass
class AcousticField:
    """Contains frequency-dependent pressures and velocity vectors of acoustic field"""
    # List of frequency-limited field components
    field: List[FrequencyLimitedField] = field(default_factory=list)
    
    def add_field(self, low_freq: float, high_freq: float, pressure: float, velocity: VelocityVectors) -> None:
        """Add a new frequency-limited field component if not exist"""
        high_band = self.get_bands(high_freq)
        low_band = self.get_bands(low_freq)
        if (low_band == None and high_band == None) or low_freq == low_band[1] or high_freq == high_band[0]:
            field_component = FrequencyLimitedField(low_freq=low_freq, high_freq=high_freq, pressure=pressure, velocity=velocity)
            self.field.append(field_component)
            print(self.field[-1])
        else:
            print(f"Frequency band already exist: ", low_band, high_band)

    def update_field(self, low_freq: float, high_freq: float, pressure: float, velocity: VelocityVectors) -> None:
        """Update field pressure and velocity vectors for a frequencies band"""
        for _band in self.field:
            if low_freq == _band.low_freq and high_freq == _band.high_freq:
               self.field[self.field.index(self.get_field(_band.low_freq, _band.high_freq))].pressure = pressure
               self.field[self.field.index(self.get_field(_band.low_freq, _band.high_freq))].velocity = velocity

    def get_bands(self, frequency) -> Tuple[float, float]:
        """Get low and high frequencies band stored from single frequency"""
        band = None
        for _band in self.field:
            if _band and _band.low_freq <= frequency <= _band.high_freq:
                band = _band
        return [band.low_freq, band.high_freq] if band else None

    def get_field(self, low_freq: float, high_freq: float) -> FrequencyLimitedField:
        """Get field pressure and velocity vectors for a frequencies band"""
        for index in range(len(self.field)):
            if low_freq == self.field[index].low_freq and high_freq == self.field[index].high_freq:
               return self.field[index]
