"""Unit tests for the scene-synthesis trajectory class."""

import numpy as np
import pytest
import scene_synthesis as ss


def test_trajectory_interval_uses_point_bounds():
    """``Trajectory.interval`` should expose the first and last point times."""
    trajectory = ss.Trajectory(points={0.5: (0.0, 0.0, 0.0), 2.0: (1.0, 0.0, 0.0), 1.0: (0.5, 0.0, 0.0)})

    np.testing.assert_allclose(trajectory.interval, np.array([0.5, 2.0]))


def test_trajectory_location_matches_sampled_points():
    """``Trajectory.location`` should pass through the sampled points."""
    trajectory = ss.Trajectory(points={0.0: (0.0, 0.0, 0.0), 1.0: (1.0, 2.0, 0.0), 2.0: (2.0, 4.0, 0.0)})

    location = np.array(trajectory.location(1.0))

    np.testing.assert_allclose(location, np.array([1.0, 2.0, 0.0]))


def test_trajectory_traj_iterates_over_requested_range():
    """``Trajectory.traj`` should iterate over positions with the requested step size."""
    trajectory = ss.Trajectory(points={0.0: (0.0, 0.0, 0.0), 1.0: (1.0, 0.0, 0.0)})

    samples = list(trajectory.traj(0.0, 1.0, 0.25))

    assert len(samples) == 4
    np.testing.assert_allclose(samples[0], (0.0, 0.0, 0.0))
    np.testing.assert_allclose(samples[-1], (0.75, 0.0, 0.0))


def test_trajectory_location_requires_at_least_two_points():
    """``Trajectory.location`` should raise a clear error for underspecified splines."""
    trajectory = ss.Trajectory(points={0.0: (0.0, 0.0, 0.0)})

    with pytest.raises(ValueError, match='at least two sampled positions'):
        trajectory.location(0.0)
