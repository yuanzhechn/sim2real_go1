"""1072-D policy elevation contract, separate from legacy 187-ray scans."""
import numpy as np
from .types import vector

HEIGHT_MAP_SPEC = {
    "frame": "base_yaw_aligned", "ordering": "xy",
    "shape_rows_y_columns_x": [16, 32], "flattening": "x_changes_fastest_then_y",
    "resolution_m": 0.1, "forward_extent_m": [0.0, 3.1],
    "lateral_extent_m": [-0.75, 0.75],
    "value_definition": "terrain_z_minus_base_z", "value_clip_m": [-1.0, 1.0],
    "invalid_fill": 0.0,
}


def elevation_observation(heights, validity):
    heights = vector(heights, 512, "terrain_heights")
    validity = vector(validity, 512, "terrain_validity")
    if not np.all(np.isfinite(heights)) or not np.all(np.isfinite(validity)):
        raise ValueError("Elevation input contains NaN/Inf")
    if not np.all((validity == 0) | (validity == 1)):
        raise ValueError("terrain_validity must contain binary 0/1 bits")
    heights = np.where(validity == 1, np.clip(heights, -1, 1), 0)
    return np.concatenate((heights, validity)).astype(np.float32)
