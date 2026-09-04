import numpy as np
import pytest

from go1_sim2real.height_scan import fit_flat_ground_scan, isaac_lab_grid


def test_isaac_grid_has_official_shape_and_xy_order():
    grid = isaac_lab_grid()
    assert grid.shape == (187, 2)
    np.testing.assert_allclose(grid[:3], [[-0.8, -0.5], [-0.7, -0.5], [-0.6, -0.5]])
    np.testing.assert_allclose(grid[17], [-0.8, -0.4])


def test_flat_ground_scan_matches_height_definition():
    grid = isaac_lab_grid()
    points = np.column_stack((grid, 0.02 * grid[:, 0] - 0.30))
    result = fit_flat_ground_scan(points, min_occupied_cells=45)
    assert result.height_scan.shape == (187,)
    assert result.rms_m < 1e-8
    np.testing.assert_allclose(result.height_scan, -points[:, 2] - 0.5, atol=1e-6)


def test_flat_ground_scan_rejects_sparse_data():
    with pytest.raises(ValueError, match="coverage|candidates"):
        fit_flat_ground_scan(np.zeros((10, 3)))
