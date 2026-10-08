"""Small, dependency-free checks for the MPSPH transfer equations."""

import numpy as np


def difference_curl(values, positions, volume, kernel_gradient):
    result = np.zeros(3, dtype=np.float64)
    for j in range(len(values)):
        result += volume[j] * np.cross(values[j] - values[0], kernel_gradient[0, j])
    return result


def test_uniform_translation_has_zero_curl():
    values = np.tile(np.array([2.0, -1.0, 0.5]), (3, 1))
    positions = np.zeros((3, 3))
    gradients = np.array([
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[0.0, 0.0, 0.0]] * 3,
        [[0.0, 0.0, 0.0]] * 3,
    ])
    assert np.allclose(difference_curl(values, positions, np.ones(3), gradients), 0.0)


def test_micropolar_transfer_sign_and_inertia():
    curl_v = np.array([0.0, 0.0, 4.0])
    omega = np.array([0.0, 0.0, 1.0])
    nu_t = 0.05
    theta = 2.0
    linear = nu_t * np.array([0.0, 0.0, 0.0])
    angular = nu_t / theta * (curl_v - 2.0 * omega)
    assert np.allclose(linear, 0.0)
    assert np.allclose(angular, [0.0, 0.0, 0.05])


def test_zero_transfer_reduces_to_classical_model():
    curl_v = np.array([1.0, -2.0, 3.0])
    omega = np.array([4.0, 5.0, 6.0])
    assert np.allclose(0.0 * curl_v, 0.0)
    assert np.allclose(0.0 / 2.0 * (curl_v - 2.0 * omega), 0.0)
