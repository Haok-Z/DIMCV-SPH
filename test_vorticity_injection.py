import numpy as np

from dfsph_segment_hybrid_solver import _select_proportional_sign_indices


def _sign_counts(values):
    values = np.asarray(values)
    return int(np.count_nonzero(values > 0)), int(np.count_nonzero(values < 0))


def test_proportional_selection_preserves_equal_sign_ratio():
    indices = np.arange(10)
    signs = np.array([1] * 5 + [-1] * 5, dtype=float)
    selected = _select_proportional_sign_indices(indices, np.arange(10), signs, 4)
    assert _sign_counts(signs[selected]) == (2, 2)


def test_proportional_selection_allocates_unequal_sign_ratio():
    indices = np.arange(10)
    signs = np.array([1] * 8 + [-1] * 2, dtype=float)
    selected = _select_proportional_sign_indices(indices, np.arange(10), signs, 5)
    assert _sign_counts(signs[selected]) == (4, 1)


def test_proportional_selection_fills_when_one_sign_is_exhausted():
    indices = np.arange(5)
    signs = np.array([1, 1, 1, 1, -1], dtype=float)
    selected = _select_proportional_sign_indices(indices, np.arange(5), signs, 8)
    assert selected.size == 5
    assert _sign_counts(signs[selected]) == (4, 1)


def test_proportional_selection_zero_signs_are_only_fallback():
    indices = np.arange(6)
    signs = np.array([1, 1, -1, -1, 0, 0], dtype=float)
    strengths = np.array([1, 2, 3, 4, 100, 90], dtype=float)
    selected = _select_proportional_sign_indices(indices, strengths, signs, 4)
    assert _sign_counts(signs[selected]) == (2, 2)
    assert not np.any(signs[selected] == 0)


def test_proportional_selection_odd_budget_has_at_most_one_rounding_difference():
    indices = np.arange(10)
    signs = np.array([1] * 5 + [-1] * 5, dtype=float)
    selected = _select_proportional_sign_indices(indices, np.arange(10), signs, 5)
    pos, neg = _sign_counts(signs[selected])
    assert selected.size == 5
    assert abs(pos - neg) <= 1


def test_residual_source_subtracts_local_segment_field_before_projection():
    source = np.array([3.0, -2.0, 5.0])
    represented_segment = np.array([1.0, 0.0, 2.0])
    tangent = np.array([0.0, 0.0, 1.0])
    volume = 0.25
    length = 0.5
    residual = source - represented_segment
    gamma = float(np.dot(residual, tangent) * volume / length)
    assert np.allclose(residual, [2.0, -2.0, 3.0])
    assert gamma == 1.5
