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
./scripts/run_post_process.py

Standalone script to run post-processing on already-rendered tracks.
"""

import os
import sys
import argparse
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physicsSolver import EntityManager
from rigidBody.lib.post_process import PostProcess, PostProcessConfig


def main():
    parser = argparse.ArgumentParser(description='Post-process rigid body audio tracks')
    parser.add_argument('config', type=str, help='Path to configuration JSON file')
    parser.add_argument('--object', type=str, default=None, 
                        help='Object name to process (default: all)')
    parser.add_argument('--denoise', type=float, default=0.7,
                        help='Spectral reduction strength (0-1)')
    parser.add_argument('--smooth', type=float, default=2.0,
                        help='Smoothing window in ms')
    parser.add_argument('--gain', type=float, default=20.0,
                        help='Maximum gain in dB')
    parser.add_argument('--mix', type=float, default=0.85,
                        help='Dry/wet mix (0=dry, 1=wet)')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Create entity manager
    entity_manager = EntityManager(args.config)
    
    # Configure post-processor
    config = PostProcessConfig(
        spectral_reduction_strength=args.denoise,
        smoothing_window_ms=args.smooth,
        max_gain_db=args.gain,
        dry_wet_mix=args.mix,
        verbose=args.verbose
    )
    
    # Create post-processor
    processor = PostProcess(
        entity_manager=entity_manager,
        config=config
    )
    
    # Process
    if args.object:
        # Process single object
        results = processor.process_object(args.object, 0)
        if args.verbose:
            print(f"Processed {args.object}: {list(results.keys())}")
    else:
        # Process all objects
        results = processor.process_all_objects()
        if args.verbose:
            for obj_name, tracks in results.items():
                print(f"Processed {obj_name}: {list(tracks.keys())}")


if __name__ == '__main__':
    main()

