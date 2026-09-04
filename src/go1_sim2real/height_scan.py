# -*- coding: utf-8 -*-
"""Isaac Lab 187-ray grid construction and guarded flat-ground fitting."""

import math

import numpy as np


class GroundPlaneResult:
    def __init__(self, height_scan, coefficients, rms_m, slope_rad, occupied_cells, total_cells):
        self.height_scan = height_scan
        self.coefficients = coefficients
        self.rms_m = rms_m
        self.slope_rad = slope_rad
        self.occupied_cells = occupied_cells
        self.total_cells = total_cells

    @property
    def coverage(self):
        return self.occupied_cells / float(self.total_cells)


def isaac_lab_grid(length_m=1.6, width_m=1.0, resolution_m=0.1):
    """Return Isaac Lab GridPatternCfg's default ``ordering='xy'`` flatten order."""
    if min(length_m, width_m, resolution_m) <= 0:
        raise ValueError("grid size/resolution must be positive")
    xs = np.arange(-length_m / 2, length_m / 2 + 1.0e-9, resolution_m)
    ys = np.arange(-width_m / 2, width_m / 2 + 1.0e-9, resolution_m)
    # torch.meshgrid(x, y, indexing="xy") produces y-major rows, x-major columns.
    return np.asarray([(x, y) for y in ys for x in xs], dtype=np.float64)


def fit_flat_ground_scan(
    points_xyz,
    height_offset_m=0.5,
    cell_size_m=0.1,
    max_rms_m=0.035,
    max_slope_rad=math.radians(15.0),
    min_occupied_cells=45,
):
    """Fit a plane to the lower envelope and extrapolate only if it is demonstrably flat.

    This is intentionally a flat-floor commissioning mode.  It must not be used as a
    replacement for measured grid cells on rough terrain.
    """
    points = np.asarray(points_xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_xyz must have shape (N, 3)")
    finite = np.all(np.isfinite(points), axis=1)
    points = points[finite]
    region = (
        (np.abs(points[:, 0]) <= 0.95)
        & (np.abs(points[:, 1]) <= 0.65)
        & (points[:, 2] >= -1.0)
        & (points[:, 2] <= 0.05)
    )
    points = points[region]
    if points.shape[0] < min_occupied_cells:
        raise ValueError("not enough finite ground candidates")

    # Obstacles are above the floor (larger z).  Keep the lowest sample per 10 cm cell.
    ix = np.floor((points[:, 0] + 0.95) / cell_size_m).astype(int)
    iy = np.floor((points[:, 1] + 0.65) / cell_size_m).astype(int)
    lower = {}
    for key, point in zip(zip(ix, iy), points):
        old = lower.get(key)
        if old is None or point[2] < old[2]:
            lower[key] = point
    samples = np.asarray(list(lower.values()), dtype=np.float64)
    if samples.shape[0] < min_occupied_cells:
        raise ValueError(
            "ground coverage too low: %d cells, require %d"
            % (samples.shape[0], min_occupied_cells)
        )

    design = np.column_stack((samples[:, 0], samples[:, 1], np.ones(samples.shape[0])))
    keep = np.ones(samples.shape[0], dtype=bool)
    coefficients = np.zeros(3)
    for _ in range(4):
        if np.count_nonzero(keep) < min_occupied_cells:
            raise ValueError("ground inliers below safety threshold")
        coefficients = np.linalg.lstsq(design[keep], samples[keep, 2], rcond=None)[0]
        residual = samples[:, 2] - design.dot(coefficients)
        center = np.median(residual[keep])
        mad = np.median(np.abs(residual[keep] - center))
        threshold = max(0.025, 3.0 * 1.4826 * mad)
        keep = np.abs(residual - center) <= threshold

    residual = samples[keep, 2] - design[keep].dot(coefficients)
    rms = float(np.sqrt(np.mean(residual * residual)))
    slope = float(math.atan(math.hypot(coefficients[0], coefficients[1])))
    if rms > max_rms_m:
        raise ValueError("ground plane RMS %.3f m exceeds %.3f m" % (rms, max_rms_m))
    if slope > max_slope_rad:
        raise ValueError("ground slope %.1f deg exceeds %.1f deg" % (math.degrees(slope), math.degrees(max_slope_rad)))

    grid = isaac_lab_grid()
    ground_z = coefficients[0] * grid[:, 0] + coefficients[1] * grid[:, 1] + coefficients[2]
    height_scan = (-ground_z - float(height_offset_m)).astype(np.float32)
    return GroundPlaneResult(
        height_scan=height_scan,
        coefficients=coefficients,
        rms_m=rms,
        slope_rad=slope,
        occupied_cells=int(np.count_nonzero(keep)),
        total_cells=int(grid.shape[0]),
    )
