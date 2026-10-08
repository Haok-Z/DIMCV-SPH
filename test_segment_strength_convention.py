"""Pure numerical checks for the physical circulation conventions.

These tests intentionally avoid importing Taichi so they can run in a minimal
Python environment and document the invariants used by the solver kernels.
"""

import numpy as np


def sph_to_segment_circulation(omega, volume, tangent, length):
    return float(np.dot(omega, tangent) * volume / length)


def test_sph_vector_to_segment_circulation_round_trip():
    omega = np.array([2.0, -1.0, 4.0])
    volume = 0.125
    tangent = np.array([0.0, 0.0, 1.0])
    length = 0.25
    gamma = sph_to_segment_circulation(omega, volume, tangent, length)
    d = length * tangent
    beta = gamma * d
    expected = np.dot(omega, tangent) * volume * tangent
    np.testing.assert_allclose(beta, expected)


def test_split_preserves_segment_circulation_vector():
    gamma = 0.7
    d = np.array([0.0, 0.0, 0.4])
    midpoint_split = 0.5 * d
    beta_before = gamma * d
    beta_after = gamma * midpoint_split + gamma * midpoint_split
    np.testing.assert_allclose(beta_after, beta_before)


def test_merge_reconstructs_circulation_from_vector_sum():
    gamma_i = 0.5
    gamma_j = -0.25
    d_i = np.array([0.2, 0.0, 0.0])
    d_j = np.array([0.0, 0.1, 0.0])
    beta = gamma_i * d_i + gamma_j * d_j
    length = 0.5 * (np.linalg.norm(d_i) + np.linalg.norm(d_j))
    tangent = beta / np.linalg.norm(beta)
    gamma_new = np.dot(beta, tangent) / length
    np.testing.assert_allclose(gamma_new * (length * tangent), beta)


def test_residual_circulation_is_a_unit_gain_one_step_correction():
    omega_loss = np.array([0.0, 3.0, 4.0])
    volume = 0.02
    tangent = omega_loss / np.linalg.norm(omega_loss)
    length = 0.1
    gamma = sph_to_segment_circulation(omega_loss, volume, tangent, length)

    represented_loss = gamma * (length * tangent)
    expected_loss = np.dot(omega_loss, tangent) * volume * tangent
    np.testing.assert_allclose(represented_loss, expected_loss)

    beta = 1.0
    np.testing.assert_allclose(beta * represented_loss, expected_loss)


def signed_circulation_alignment(gamma_i, d_i, gamma_j, d_j):
    wi = gamma_i * np.asarray(d_i, dtype=float)
    wj = gamma_j * np.asarray(d_j, dtype=float)
    return float(np.dot(wi, wj) / (np.linalg.norm(wi) * np.linalg.norm(wj)))


def accepts_opposing_merge(alignment, angle_tolerance):
    return alignment <= -np.cos(angle_tolerance)


def test_merge_direction_accepts_opposing_signed_circulation():
    alignment = signed_circulation_alignment(1.0, [1.0, 0.0, 0.0], 1.0, [-1.0, 0.0, 0.0])
    assert accepts_opposing_merge(alignment, np.deg2rad(60.0))


def test_merge_direction_rejects_same_signed_circulation():
    alignment = signed_circulation_alignment(1.0, [1.0, 0.0, 0.0], 1.0, [1.0, 0.0, 0.0])
    assert not accepts_opposing_merge(alignment, np.deg2rad(60.0))


def test_zero_residual_blend_preserves_existing_gamma_before_decay():
    existing_gamma = 2.5
    residual_segment_gamma_blend = 0.0
    target_gamma = -7.0
    updated = (
        (1.0 - residual_segment_gamma_blend) * existing_gamma
        + residual_segment_gamma_blend * target_gamma
    )
    np.testing.assert_allclose(updated * 0.98, existing_gamma * 0.98)
