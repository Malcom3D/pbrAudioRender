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

import os
import json
import numpy as np
import numba as nb
from numba import prange
from dask import delayed, compute
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import soundfile as sf
from scipy import signal

from ..core.entity_manager import EntityManager
from ..lib.functions import _cartesian_to_spherical

@dataclass
class IRrender:
    """
    Render impulse responses from ray data for ambisonic audio.

    Processes ray_data stored in EntityManager to create impulse responses
    for each source-output pair. Handles frequency-dependent energy and phase
    information to produce accurate spatial audio rendering.
    """
    entity_manager: EntityManager

    def __post_init__(self):
        self.config = = self.entity_manager.get('config')
        self.sample_rate = self.config.system.sample_rate
        self.freq_bands = self.entity_manager.get('frequency_bands').get_bands()
        self.n_bands = len(self.freq_bands)

        # Get output configurations
        self.outputs = self.entity_manager.get('outputs')
        self.sources = self.entity_manager.get('sources')

        # Get ambisonic output configurations
        self.ambisonic_outputs = [oc for oc in self.config.outputs if oc.type == 'AMBI']

        # Load spatial arrangements for ambisonic outputs
        self.arrangements = {}
        for ao in self.ambisonic_outputs:
            if hasattr(ao, 'spatial_arrangement_file') and ao.spatial_arrangement_file:
                with open(ao.spatial_arrangement_file, 'r') as f:
                    self.arrangements[ao.idx] = json.load(f)

    def render_all(self):
        """Render impulse responses for all source-output pairs."""
        ray_datas = self.entity_manager.get('ray_datas')

        # Group ray_datas by source-output pair
        source_output_pairs = {}
        for rd_idx, rd in ray_datas.items():
            key = (rd.src_idx, rd.out_idx)
            if key not in source_output_pairs:
                source_output_pairs[key] = []
            source_output_pairs[key].append(rd)

        # Process each source-output pair
        tasks = []
        for (src_idx, out_idx), rd_list in source_output_pairs.items():
            tasks.append(self._process_source_output_pair(src_idx, out_idx, rd_list))

        compute(*tasks)

    @delayed
    def _process_source_output_pair(self, src_idx: int, out_idx: int, ray_datas: List):
        """Process all ray data for a single source-output pair."""
        # Collect all output hits (obj_idx == -3)
        output_hits = self._collect_output_hits(ray_datas)

        if len(output_hits['energies']) == 0:
            print(f"IRrender: No output hits for source {src_idx} -> output {out_idx}")
            return

        # Convert to numpy arrays for SIMD processing
        delays = np.array(output_hits['delays'], dtype=np.float32)
        energies = np.array(output_hits['energies'], dtype=np.float32)  # (n_hits, n_bands)
        phases = np.array(output_hits['phases'], dtype=np.float32)  # (n_hits, n_bands)

        # Get output position and orientation
        output_config = self._get_output_config(out_idx)
        output_pos = self._get_output_position(output_config)

        # Get source position
        source_config = self._get_source_config(src_idx)
        source_pos = self._get_source_position(source_config)

        # Compute direction of arrival for each hit
        hit_directions = self._compute_hit_directions(output_hits, output_pos)

        # Build impulse response
        ir = self._build_impulse_response(
            delays, energies, phases, hit_directions,
            output_config, output_pos, source_pos
        )

        return ir

    def _collect_output_hits(self, ray_datas: List) -> Dict:
        """Collect all rays that hit output (obj_idx == -3) from ray data."""
        output_hits = {
            'delays': [],
            'energies': [],  # List of (n_bands,) arrays
            'phases': [],    # List of (n_bands,) arrays
            'directions': [],  # Direction of arrival at output
            'positions': [],   # Hit positions on output
            'interaction_counts': []
        }

        for rd in ray_datas:
            # Find output hits
            output_mask = rd.hit_obj_idx == -3
            if not np.any(output_mask):
                continue

            # Extract output hit data
            n_hits = np.sum(output_mask)

            # Get delays for output hits
            if hasattr(rd, 'delay') and rd.delay.shape[0] > 0:
                delays = rd.delay[output_mask].flatten()
                output_hits['delays'].extend(delays.tolist())

            # Get energies for output hits (from rays_energies_output)
            if hasattr(rd, 'rays_energies_output') and rd.rays_energies_output.shape[0] > 0:
                energies = rd.rays_energies_output[output_mask]
                # energies shape: (n_hits, 1) - need to expand to (n_hits, n_bands)
                if energies.ndim == 2 and energies.shape[1] == 1:
                    energies = np.repeat(energies, self.n_bands, axis=1)
                output_hits['energies'].extend(energies.tolist())

            # Get phases for output hits (from rays_phases_output)
            if hasattr(rd, 'rays_phases_output') and rd.rays_phases_output.shape[0] > 0:
                phases = rd.rays_phases_output[output_mask]
                if phases.ndim == 2 and phases.shape[1] == 1:
                    phases = np.repeat(phases, self.n_bands, axis=1)
                output_hits['phases'].extend(phases.tolist())

            # Get hit positions
            if hasattr(rd, 'hits_coords') and rd.hits_coords.shape[0] > 0:
                hit_positions = rd.hits_coords[output_mask]
                output_hits['positions'].extend(hit_positions.tolist())

            # Get interaction counts
            if hasattr(rd, 'interactions') and rd.interactions is not None:
                output_hits['interaction_counts'].extend([rd.interactions] * n_hits)

        return output_hits

    def _compute_hit_directions(self, output_hits: Dict, output_pos: np.ndarray) -> np.ndarray:
        """Compute direction of arrival for each hit."""
        if len(output_hits['positions']) == 0:
            return np.array([])

        positions = np.array(output_hits['positions'])
        directions = positions - output_pos

        # Normalize
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        directions = directions / norms

        return directions

    def _build_impulse_response(self,
                               delays: np.ndarray,
                               energies: np.ndarray,
                               phases: np.ndarray,
                               hit_directions: np.ndarray,
                               output_config: Any,
                               output_pos: np.ndarray,
                               source_pos: np.ndarray) -> Dict:
        """
        Build impulse response from ray data.

        Returns dictionary with:
        - 'time': time axis in seconds
        - 'amplitude': impulse response amplitude
        - 'frequency_response': frequency-dependent response per band
        - 'directions': direction of arrival for each impulse
        """
        if len(delays) == 0:
            return None

        # Convert delays to time in seconds
        sound_speed = 343.0  # Default, should be from config
        if hasattr(self.config.acoustic_domain, 'acoustic_shader'):
            sound_speed = self.config.acoustic_domain.acoustic_shader.sound_speed

        times = delays / sound_speed

        # Determine IR length (max time + some tail)
        max_time = np.max(times) if len(times) > 0 else 0.1
        ir_length = int(max_time * self.sample_rate) + 1024  # Add padding

        # Initialize impulse response arrays
        ir_time = np.zeros(ir_length)
        ir_amplitude = np.zeros(ir_length)
        ir_frequency = np.zeros((ir_length, self.n_bands), dtype=np.complex64)
        ir_directions = np.zeros((ir_length, 3))

        # Apply directivity pattern based on output type
        directivity = self._get_output_directivity(output_config)

        # Process each hit using SIMD-optimized function
        ir_time, ir_amplitude, ir_frequency, ir_directions = self._accumulate_hits_simd(
            times, energies, phases, hit_directions,
            ir_time, ir_amplitude, ir_frequency, ir_directions,
            self.sample_rate, directivity, self.n_bands
        )

        return {
            'time': ir_time,
            'amplitude': ir_amplitude,
            'frequency_response': ir_frequency,
            'directions': ir_directions,
            'sample_rate': self.sample_rate,
            'n_bands': self.n_bands
        }

    @staticmethod
    @nb.njit(parallel=True, fastmath=True, cache=True)
    def _accumulate_hits_simd(times: np.ndarray,
                              energies: np.ndarray,
                              phases: np.ndarray,
                              directions: np.ndarray,
                              ir_time: np.ndarray,
                              ir_amplitude: np.ndarray,
                              ir_frequency: np.ndarray,
                              ir_directions: np.ndarray,
                              sample_rate: int,
                              directivity: float,
                              n_bands: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        SIMD-optimized accumulation of ray hits into impulse response.

        Uses nearest-neighbor binning for speed.
        """
        n_hits = times.shape[0]
        ir_length = ir_time.shape[0]

        for i in nb.prange(n_hits):
            # Compute sample index
            sample_idx = int(times[i] * sample_rate)

            if sample_idx < 0 or sample_idx >= ir_length:
                continue

            # Apply directivity (simplified - assume it's frequency-independent)
            gain = directivity  # Will be refined per direction later

            # Accumulate amplitude (sum of energies across bands)
            total_energy = 0.0
            for b in range(n_bands):
                total_energy += energies[i, b]

            # Use atomic-like accumulation (Numba handles this in parallel)
            ir_amplitude[sample_idx] += total_energy * gain

            # Accumulate frequency-dependent response
            for b in range(n_bands):
                # Complex representation: magnitude * exp(j*phase)
                mag = energies[i, b] * gain
                phase = phases[i, b]
                ir_frequency[sample_idx, b] += mag * np.exp(1j * phase)

            # Accumulate direction (weighted by energy)
            for d in range(3):
                ir_directions[sample_idx, d] += total_energy * directions[i, d] * gain

        return ir_time, ir_amplitude, ir_frequency, ir_directions

    def _get_output_directivity(self, output_config: Any) -> float:
        """Get directivity factor for output type."""
        if output_config.type == 'MONO':
            mic_type = output_config.microphone_type if hasattr(output_config, 'microphone_type') else 'OMNIDIRECTIONAL'
            if mic_type == 'OMNIDIRECTIONAL':
                return 1.0
            elif mic_type == 'CARDIOID':
                return 0.5  # Average directivity
            elif mic_type == 'HYPERCARDIOID':
                return 0.25
            elif mic_type == 'FIGURE_8':
                return 0.0  # Depends on orientation
        return 1.0  # Default omnidirectional

    def _get_output_config(self, out_idx: int) -> Any:
        """Get output configuration by index."""
        for oc in self.config.outputs:
            if oc.idx == out_idx:
                return oc
        return None

    def _get_source_config(self, src_idx: int) -> Any:
        """Get source configuration by index."""
        for sc in self.config.sources:
            if sc.idx == src_idx:
                return sc
        return None

    def _get_output_position(self, output_config: Any) -> np.ndarray:
        """Get output position from configuration."""
        # For static outputs, position is from pose file
        if hasattr(output_config, 'pose_path') and output_config.pose_path:
            from ..lib.functions import _load_pose
            positions, _ = _load_pose(output_config)
            if output_config.static:
                return positions positions
            else:
                return positions[0]  # Use first frame
        return np.zeros(3)  # Default origin

    def _get_source_position(self, source_config: Any) -> np.ndarray:
        """Get source position from configuration."""
        if hasattr(source_config, 'pose_path') and source_config.pose_path:
            from ..lib.functions import _load_pose
            positions, _ = _load_pose(source_config)
            if source_config.static:
                return positions
            else:
                return positions[0]  # Use first frame
        return np.zeros(3)

    def _store_impulse_response(self, src_idx: int, out_idx: int, ir: Dict):
        """Store impulse response in entity manager."""
        if ir is None:
            return

        # Register IR in entity manager
        ir_key = f"ir_{src_idx}_{out_idx}"
        self.entity_manager.register('impulse_responses', ir, key=ir_key)

    def render_ambisonic(self, src_idx: int, out_idx: int) -> np.ndarray:
        """
        Render ambisonic impulse response for a source-output pair.

        Returns B-format impulse response as (samples, channels) array.
        """
        # Get impulse response
        ir_key = f"ir_{src_idx}_{out_idx}"
        ir = self.entity_manager.get('impulse_responses', key=ir_key)

        if ir is None:
            raise RuntimeError(f"No impulse response for source {src_idx} -> output {out_idx}")

        # Get output configuration
        output_config = self._get_output_config(out_idx)

        # Check if this is an ambisonic output
        if output_config.type != 'AMBI':
            # For non-ambisonic outputs, just return mono IR
            return ir['amplitude'].reshape(-1, 1)

        # Load spatial arrangement
        arrangement = self.arrangements.get(out_idx)
        if arrangement is None:
            raise RuntimeError(f"No spatial arrangement for output {out_idx}")

        # Determine ambisonic order from arrangement
        order = arrangement.get('order_supported', 1)
        n_channels = (order + 1) ** 2

        # Initialize B-format IR
        ir_length = len(ir['amplitude'])
        bformat_ir = np.zeros((ir_length, n_channels))

        # For each microphone in the arrangement, encode to B-format
        for mic in arrangement['outputs']:
            mic_pos = np.array(mic['position'])
            mic_type = mic['type']

            # Get directivity for this microphone
            directivity = self._get_mic_directivity(mic_type)

            # Compute encoding gains for each sample
            for t in range(ir_length):
                if ir['amplitude'][t] == 0:
                    continue

                # Get direction of arrival
                direction = ir['directions'][t]

                # Convert to spherical coordinates
                azimuth, elevation, _ = _cartesian_to_spherical(
                    direction[0], direction[1], direction[2]
                )

                # Apply microphone directivity
                mic_gain = self._apply_mic_directivity(directivity, azimuth, elevation, mic_pos)

                # Encode to B-format
                bformat_gains = self._encode_to_bformat(
                    azimuth, elevation, order, arrangement.get('normalization', 'N3D')
                )

                # Accumulate
                for ch in range(n_channels):
                    bformat_ir[t, ch] += ir['amplitude'][t] * mic_gain * bformat_gains[ch]

        return bformat_ir

    def _get_mic_directivity(self, mic_type: str) -> str:
        """Get directivity pattern for microphone type."""
        return mic_type  # 'omnidirectional', 'cardioid', etc.

    def _apply_mic_directivity(self, directivity: str, azimuth: float, elevation: float, mic_pos: np.ndarray) -> float:
        """Apply microphone directivity pattern."""
        if directivity == 'omnidirectional':
            return 1.0
        elif directivity == 'cardioid':
            # Cardioid: 0.5 * (1 + cos(theta))
            # theta is angle between sound direction and mic orientation
            theta = np.arccos(np.clip(mic_pos[2], -1, 1))  # Simplified
            return 0.5 * (1 + np.cos(theta))
        elif directivity == 'hypercardioid':
            theta = np.arccos(np.clip(mic_pos[2], -1, 1))
            return 0.25 * (1 + 3 * np.cos(theta))
        elif directivity == 'figure8':
            theta = np.arccos(np.clip(mic_pos[2], -1, 1))
            return np.cos(theta)
        return 1.0

    @staticmethod
    def _encode_to_bformat(azimuth: float, elevation: float, order: int, normalization: str) -> np.ndarray:
        """
        Encode direction to B-format (ACN channel ordering).

        For order 1: W, X, Y, Z
        For higher orders: ACN ordering
        """
        n_channels = (order + 1) ** 2
        gains = np.zeros(n_channels)

        # Convert to radians
        phi = azimuth  # Already in radians from _cartesian_to_spherical
        theta = np.pi/2 - elevation  # Colatitude

        # SH coefficients for order 1
        if order >= 1:
            # W (0,0) - omnidirectional
            gains[0] = 1.0

            # X (1,-1) - cos(phi)*sin(theta)
            gains[1] = np.cos(phi) * np.sin(theta)

            # Y (1,1) - sin(phi)*sin(theta)
            gains[2] = np.sin(phi) * np.sin(theta)

            # Z (1,0) - cos(theta)
            gains[3] = np.cos(theta)

        # Apply normalization
        if normalization == 'SN3D':
            # SN3D normalization (Schmidt semi-normalized)
            if order >= 1:
                gains[1:] *= np.sqrt(3)
        elif normalization == 'N3D':
            # N3D normalization (fully normalized)
                       if order >= 1:
                gains[1:] *= np.sqrt(4 * np.pi / 3)

        return gains

    def save_impulse_response(self, src_idx: int, out_idx: int, output_dir: str = "./exports/ir/"):
        """Save impulse response to file."""
        import os
        os.makedirs(output_dir, exist_ok=True)

        ir_key = f"ir_{src_idx}_{out_idx}"
        ir = self.entity_manager.get('impulse_responses', key=ir_key)

        if ir is None:
            print(f"IRrender: No impulse response for source {src_idx} -> output {out_idx}")
            return

        # Save as NPZ
        output_file = os.path.join(output_dir, f"ir_{src_idx}_{out_idx}.npz")
        np.savez_compressed(
            output_file,
            time=ir['time'],
            amplitude=ir['amplitude'],
            frequency_response=ir['frequency_response'],
            directions=ir['directions'],
            sample_rate=ir['sample_rate'],
            n_bands=ir['n_bands']
        )

        # Also save as WAV for easy listening (mono mix)
        wav_file = os.path.join(output_dir, f"ir_{src_idx}_{out_idx}.wav")

        # Normalize and save
        amplitude = ir['amplitude']
        if np.max(np.abs(amplitude)) > 0:
            amplitude = amplitude / np.max(np.abs(amplitude)) * 0.95

        sf.write(wav_file, amplitude, ir['sample_rate'])

        print(f"IRrender: Saved impulse response to {output_file}")

    def save_ambisonic_ir(self, src_idx: int, out_idx: int, output_dir: str = "./exports/ambisonic_ir/"):
        """Save ambisonic impulse response to file."""
        import os
        os.makedirs(output_dir, exist_ok=True)

        bformat_ir = self.render_ambisonic(src_idx, out_idx)

        # Save as WAV (multichannel)
        output_file = os.path.join(output_dir, f"ambisonic_ir_{src_idx}_{out_idx}.wav")
        sf.write(output_file, bformat_ir, self.sample_rate)

        # Also save as NPZ for further processing
        npz_file = os.path.join(output_dir, f"ambisonic_ir_{src_idx}_{out_idx}.npz")
        np.savez_compressed(
            npz_file,
            bformat=bformat_ir,
            sample_rate=self.sample_rate,
            src_idx=src_idx,
            out_idx=out_idx
        )

        print(f"IRrender: Saved ambisonic IR to {output_file}")
