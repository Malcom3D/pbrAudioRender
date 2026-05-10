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

from scipy.interpolate import CubicSpline, PchipInterpolator

from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field

from ..core.entity_manager import EntityManager
from ..lib.output_data import OutputData
from ..lib.functions import _cartesian_to_spherical

@dataclass
class BandInterpolatorSet:
    """
    Holds interpolators for a single frequency band across all frames.
    Optimized for CPU SIMD operations.
    """
    band_idx: int
    frame_times: np.ndarray  # (n_frames,) - time in seconds for each frame
    energy_interpolator: Optional[CubicSpline] = None
    phase_interpolator: Optional[CubicSpline] = None
    azimuth_interpolator: Optional[PchipInterpolator] = None
    elevation_interpolator: Optional[PchipInterpolator] = None
    delay_interpolator: Optional[CubicSpline] = None
    
    def interpolate_at_time(self, time: float) -> Dict[str, float]:
        """
        Interpolate all values at a given time.
        
        Args:
            time: Time in seconds
            
        Returns:
            Dictionary with interpolated values
        """
        result = {}
        
        if self.energy_interpolator is not None:
            result['energy'] = float(self.energy_interpolator(time))
        if self.phase_interpolator is not None:
            result['phase'] = float(self.phase_interpolator(time))
        if self.azimuth_interpolator is not None:
            result['azimuth'] = float(self.azimuth_interpolator(time))
        if self.elevation_interpolator is not None:
            result['elevation'] = float(self.elevation_interpolator(time))
        if self.delay_interpolator is not None:
            result['delay'] = float(self.delay_interpolator(time))
            
        return result
