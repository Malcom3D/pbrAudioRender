import sys, os
from pbrAudioCommon.lib.import_helper import np
sys.path.append(os.getcwd())
from typing import List, Dict, Any, Optional, Tuple, Union

from utils.config import Config
from core.soxel import Soxel, SoxelGrid

current_frame = 0
config_file = 'config.json'
config = Config(config_file)

ad_geometry = config.acoustic_domain.geometry
ad_geometry = ad_geometry if isinstance(ad_geometry, np.ndarray) else np.array(ad_geometry)
shape_z = (np.linalg.norm(ad_geometry[0] - ad_geometry[1] / config.acoustic_domain.voxel_size).astype(int))
shape_y = (np.linalg.norm(ad_geometry[1] - ad_geometry[2]) / config.acoustic_domain.voxel_size).astype(int)
shape_x = (np.linalg.norm(ad_geometry[2] - ad_geometry[6]) / config.acoustic_domain.voxel_size).astype(int)
config.acoustic_domain.shape = [int(shape_x), int(shape_y), int(shape_z)]

soxel_grid = SoxelGrid(config)


band = soxel_grid.soxels[28, 40, 10].input_pressures.get_bands(732)
low = band[0][0]
high = band[0][1]
field = soxel_grid.soxels[28, 40, 10].input_pressures.get_field(low, high)
field

soxel_grid.current_frame = 23

band = soxel_grid.soxels[28, 40, 10].input_pressures.get_bands(732)
low = band[0][0]
high = band[0][1]
field = soxel_grid.soxels[28, 40, 10].input_pressures.get_field(low, high)
field
