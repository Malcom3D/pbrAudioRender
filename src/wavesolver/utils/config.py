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

import yaml
import json
from typing import Dict, Any, Optional
import os

class SimulationConfig:
    """Configuration management for acoustic wave simulations."""
    
    def __init__(self, config_file: Optional[str] = None):
        self.default_config = self._get_default_config()
        self.config = self.default_config.copy()
        
        if config_file and os.path.exists(config_file):
            self.load_config(config_file)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default simulation configuration."""
        return {
            'simulation': {
                'dimensions': [64, 64, 64],
                'voxel_size': 0.1,
                'sound_speed': 343.0,
                'density': 1.2,
                'cfl_number': 0.3,
                'duration': 1.0,
                'sample_rate': 48000
            },
            'solver': {
                'boundary_type': 'absorbing',
                'boundary_strength': 0.1,
                'absorption_enabled': True
            },
            'sources': {
                'canned_sources': [],
                'source_amplitude': 1.0
            },
            'output': {
                'ambisonic_order': 1,
                'output_positions': [],
                'export_openvdb': True,
                'openvdb_export_interval': 10,
                'zarr_store_path': 'simulation_data.zarr'
            },
            'rendering': {
                'listener_trajectory': [],
                'distance_attenuation': True
            }
        }
    
    def load_config(self, config_file: str):
        """Load configuration from YAML or JSON file."""
        try:
            with open(config_file, 'r') as f:
                if config_file.endswith('.yaml') or config_file.endswith('.yml'):
                    loaded_config = yaml.safe_load(f)
                elif config_file.endswith('.json'):
                    loaded_config = json.load(f)
                else:
                    raise ValueError("Config file must be YAML or JSON")
            
            # Deep merge with existing config
            self._deep_merge(self.config, loaded_config)
            print(f"Loaded configuration from {config_file}")
            
        except Exception as e:
            print(f"Error loading config file: {str(e)}")
            print("Using default configuration")
    
    def save_config(self, config_file: str):
        """Save current configuration to file."""
        try:
            with open(config_file, 'w') as f:
                if config_file.endswith('.yaml') or config_file.endswith('.yml'):
                    yaml.dump(self.config, f, default_flow_style=False)
                elif config_file.endswith('.json'):
                    json.dump(self.config, f, indent=2)
                else:
                    raise ValueError("Config file must be YAML or JSON")
            
            print(f"Saved configuration to {config_file}")
            
        except Exception as e:
            print(f"Error saving config file: {str(e)}")
    
    def _deep_merge(self, base: Dict[str, Any], update: Dict[str, Any]):
        """Recursively merge two dictionaries."""
        for key, value in update.items():
            if (key in base and isinstance(base[key], dict) and 
                isinstance(value, dict)):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation."""
        keys = key.split('.')
        current = self.config
        
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default
        
        return current
    
    def set(self, key: str, value: Any):
        """Set configuration value using dot notation."""
        keys = key.split('.')
        current = self.config
        
        for k in keys[:-1]:
            if k not in current or not isinstance(current[k], dict):
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate the current configuration."""
        errors = []
        
        # Simulation parameters
        sim = self.config['simulation']
        if not all(d > 0 for d in sim['dimensions']):
            errors.append("All simulation dimensions must be positive")
        
        if sim['voxel_size'] <= 0:
            errors.append("Voxel size must be positive")
        
        if sim['sound_speed'] <= 0:
            errors.append("Sound speed must be positive")
        
        if sim['cfl_number'] <= 0 or sim['cfl_number'] > 1.0:
            errors.append("CFL number must be between 0 and 1")
        
        # Output parameters
        output = self.config['output']
        if output['ambisonic_order'] < 0 or output['ambisonic_order'] > 3:
            errors.append("Ambisonic order must be between 0 and 3")
        
        return len(errors) == 0, errors
    
    def add_sound_source(self, position: Tuple[float, float, float], 
                        wav_file: str, 
                        amplitude: float = 1.0):
        """Add a sound source to configuration."""
        source = {
            'position': list(position),
            'wav_file': wav_file,
            'amplitude': amplitude
        }
        
        if 'canned_sources' not in self.config['sources']:
            self.config['sources']['canned_sources'] = []
        
        self.config['sources']['canned_sources'].append(source)
    
    def add_output_position(self, position: Tuple[float, float, float]):
        """Add an output position for audio rendering."""
        if 'output_positions' not in self.config['output']:
            self.config['output']['output_positions'] = []
        
        self.config['output']['output_positions'].append(list(position))
    
    def add_listener_position(self, position: Tuple[float, float, float]):
        """Add a listener position for audio rendering."""
        if 'listener_trajectory' not in self.config['rendering']:
            self.config['rendering']['listener_trajectory'] = []
        
        self.config['rendering']['listener_trajectory'].append(list(position))
    
    def get_simulation_parameters(self) -> Dict[str, Any]:
        """Get simulation parameters for solver initialization."""
        sim = self.config['simulation']
        return {
            'dimensions': tuple(sim['dimensions']),
            'voxel_size': sim['voxel_size'],
            'sound_speed': sim['sound_speed'],
            'density': sim['density'],
            'cfl_number': sim['cfl_number']
        }
    
    def __str__(self) -> str:
        """String representation of configuration."""
        return json.dumps(self.config, indent=2)
