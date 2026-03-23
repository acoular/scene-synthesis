"""Unit tests for directivity models.

These tests exercise the :mod:`scene_synthesis.directivities` module
so that the base :class:`Directivity` API and the concrete omni/cardioid
patterns are covered.
"""

import numpy as np
from scene_synthesis.directivities import CardioidDirectivity, Directivity, OmniDirectivity


def test_directivity_base_raises_not_implemented_error():
    """Base ``Directivity`` must not be usable without overriding.

    The default implementation of :meth:`Directivity.get_coefficients`
    should raise :class:`NotImplementedError` to fail fast when the base
    class is used directly.
    """
    directivity = Directivity()

    orientation = np.eye(3)
    target_directions = np.eye(3)

    msg = 'Directivity.get_coefficients must be implemented by subclasses'

    import pytest

    with pytest.raises(NotImplementedError, match=msg):
        directivity.get_coefficients(orientation, target_directions)


def test_omnidirectivity_returns_ones():
    """``OmniDirectivity`` should return ones for all target directions."""
    directivity = OmniDirectivity()

    orientation = np.eye(3)
    # three arbitrary unit directions as columns
    target_directions = np.array(
        [
            [1.0, 0.0, 0.0],  # +x
            [0.0, 1.0, 0.0],  # +y
            [0.0, 0.0, 1.0],  # +z
        ]
    ).T

    coeffs = directivity.get_coefficients(orientation, target_directions)

    assert coeffs.shape == (target_directions.shape[1],)
    assert np.allclose(coeffs, np.ones_like(coeffs))


def test_cardioid_directivity_matches_analytic_pattern():
    """``CardioidDirectivity`` follows 0.5 * (1 + cos(theta)).

    With orientation ``I`` and target directions given as unit vectors,
    the implementation reduces to the standard cardioid pattern based on
    the angle with the forward axis ``orientation[2]``.
    """
    directivity = CardioidDirectivity()

    orientation = np.eye(3)
    forward = orientation[2]

    # Forward, side, and backward directions as columns
    target_directions = np.array(
        [
            [0.0, 0.0, 1.0],   # forward
            [1.0, 0.0, 0.0],   # side
            [0.0, 0.0, -1.0],  # backward
        ]
    ).T

    coeffs = directivity.get_coefficients(orientation, target_directions)

    # Analytical cardioid: 0.5 * (1 + cos(theta)) where cos(theta) = dot(fwd, dir)
    cos_thetas = forward @ target_directions
    expected = 0.5 * (1.0 + cos_thetas)

    assert coeffs.shape == expected.shape
    assert np.allclose(coeffs, expected)
