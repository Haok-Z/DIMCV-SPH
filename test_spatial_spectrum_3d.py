"""Focused numerical tests for the offline spectrum utilities.

Run with: python -m unittest test_spatial_spectrum_3d.py
"""
from __future__ import annotations

import unittest

import numpy as np

from spatial_spectrum_3d import (
    common_valid_mask,
    prepare_masked_field,
    radial_shell_spectrum,
    reconstruct_sph_vector_grid,
    vector_curl_spectral,
)


class SpatialSpectrumTests(unittest.TestCase):
    def setUp(self):
        self.size = 32
        self.lo = np.zeros(3)
        self.hi = np.ones(3)
        self.spacing = np.ones(3) / self.size

    def test_parseval_and_single_mode_peak(self):
        coordinate = (np.arange(self.size) + 0.5) / self.size
        x, _, _ = np.meshgrid(coordinate, coordinate, coordinate, indexing="ij")
        field = np.zeros((self.size, self.size, self.size, 3))
        field[..., 1] = np.sin(2.0 * np.pi * 4.0 * x)
        prepared, _ = prepare_masked_field(field, np.ones(field.shape[:3], dtype=bool), alpha=0.0)
        radial, metadata = radial_shell_spectrum(prepared, self.spacing)
        self.assertAlmostEqual(metadata["total_energy"], metadata["physical_space_energy"], places=10)
        peak = radial[np.argmax(radial[:, 1]), 0]
        shell_width = radial[1, 0] - radial[0, 0]
        self.assertLess(abs(peak - 2.0 * np.pi * 4.0), shell_width)

    def test_constant_field_has_no_fluctuation_energy(self):
        field = np.ones((self.size, self.size, self.size, 3))
        prepared, _ = prepare_masked_field(field, np.ones(field.shape[:3], dtype=bool), alpha=0.0)
        radial, metadata = radial_shell_spectrum(prepared, self.spacing)
        self.assertAlmostEqual(float(radial[:, 1].sum()), 0.0, places=12)
        self.assertEqual(metadata["total_energy"], 0.0)

    def test_spectral_curl(self):
        coordinate = (np.arange(self.size) + 0.5) / self.size
        x, _, _ = np.meshgrid(coordinate, coordinate, coordinate, indexing="ij")
        field = np.zeros((self.size, self.size, self.size, 3))
        field[..., 1] = np.sin(2.0 * np.pi * 3.0 * x)
        curl = vector_curl_spectral(field, self.spacing)
        expected = 2.0 * np.pi * 3.0 * np.cos(2.0 * np.pi * 3.0 * x)
        np.testing.assert_allclose(curl[..., 2], expected, rtol=1e-10, atol=1e-10)

    def test_sph_reconstruction_and_common_mask(self):
        grid, weight = reconstruct_sph_vector_grid(
            np.array([[0.5, 0.5, 0.5]]), np.array([[1.0, 2.0, 3.0]]),
            self.lo, self.hi, (16, 16, 16), 0.1,
        )
        self.assertGreater(weight[8, 8, 8], 0.0)
        np.testing.assert_allclose(grid[8, 8, 8], [1.0, 2.0, 3.0])
        other = weight.copy()
        other[8, 8, 8] = 0.0
        mask = common_valid_mask((weight, other))
        self.assertFalse(mask[8, 8, 8])


if __name__ == "__main__":
    unittest.main()
