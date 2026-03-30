"""Unit tests for the scene-synthesis environment class."""

import numpy as np
import scene_synthesis as ss


def test_environment_apparent_r_matches_euclidean_distance():
    """``Environment.apparent_r`` should match Euclidean distance in free field."""
    environment = ss.Environment(c=343.0)

    spos = np.array([[1.0], [0.0], [0.0]])
    mpos = np.array([[0.0], [0.0], [0.0]])

    distances = environment.apparent_r(spos, mpos)

    np.testing.assert_allclose(distances, np.array([1.0]))


def test_environment_spread_is_inverse_distance():
    """``Environment.spread`` should implement spherical spreading ``1 / r``."""
    environment = ss.Environment(c=343.0)

    spos = np.array([[2.0], [0.0], [0.0]])
    mpos = np.array([[0.0], [0.0], [0.0]])

    spread = environment.spread(spos, mpos)

    np.testing.assert_allclose(spread, np.array([0.5]))


def test_environment_accepts_single_microphone_position_as_1d_array():
    """``Environment._r`` should accept a single microphone position as ``(3,)``."""
    environment = ss.Environment(c=343.0)

    spos = np.array([[1.0], [0.0], [0.0]])
    mpos = np.array([0.0, 0.0, 0.0])

    distances = environment.apparent_r(spos, mpos)

    np.testing.assert_allclose(distances, np.array([1.0]))


def test_environment_spread_supports_multiple_microphones():
    """``Environment.spread`` should broadcast over multiple microphone positions."""
    environment = ss.Environment(c=343.0)

    spos = np.array([[1.0], [0.0], [0.0]])
    mpos = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ]
    )

    spread = environment.spread(spos, mpos)
    expected = np.array([[1.0, 1.0 / np.sqrt(2.0)]])

    np.testing.assert_allclose(spread, expected)
