from pathlib import Path

import numpy as np

from gnnsr.io import memory_map_npz, npz_member_info


def test_memory_map_uncompressed_npz(tmp_path: Path):
    expected = np.arange(30, dtype=np.float64).reshape(5, 6)
    path = tmp_path / "data.npz"
    np.savez(path, values=expected)
    info = npz_member_info(path, "values")
    mapped = memory_map_npz(path, "values")
    assert info.shape == expected.shape
    np.testing.assert_array_equal(mapped, expected)
    np.testing.assert_array_equal(mapped[3, [1, 4]], expected[3, [1, 4]])


def test_compressed_npz_is_rejected(tmp_path: Path):
    path = tmp_path / "compressed.npz"
    np.savez_compressed(path, values=np.arange(5))
    try:
        memory_map_npz(path, "values")
    except ValueError as error:
        assert "compressed" in str(error)
    else:
        raise AssertionError("Expected compressed archive to be rejected")
