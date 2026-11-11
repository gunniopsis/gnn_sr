import numpy as np
import pytest

from gnnsr.baseline import project_parent_wse_to_fine_depth


def test_parent_projection_inherits_wse_and_uses_fine_elevation():
    coarse_wse = np.array([5.5, 7.0])
    fine_parent = np.array([0, 0, 1, 1])
    fine_elevation = np.array([5.0, 5.49, 6.5, 7.1])

    wse, depth = project_parent_wse_to_fine_depth(
        coarse_wse, fine_parent, fine_elevation, dry_threshold_m=0.03
    )

    np.testing.assert_allclose(wse, [5.5, 5.49, 7.0, 7.1])
    np.testing.assert_allclose(depth, [0.5, 0.0, 0.5, 0.0])


def test_parent_projection_rejects_invalid_parent_index():
    with pytest.raises(ValueError, match="outside"):
        project_parent_wse_to_fine_depth(
            np.array([5.0]), np.array([1]), np.array([4.0]), dry_threshold_m=0.03
        )
