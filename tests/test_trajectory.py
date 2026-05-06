"""Unit tests for the scene-synthesis trajectory classes."""

import numpy as np
import pytest
import scene_synthesis as ss


def test_trajectory_accepts_independent_location_and_velocity_functions():
    """``Trajectory`` should allow independently defined position and velocity."""
    trajectory = ss.Trajectory(
        location=lambda t: np.array([t, 0.0 * t, 1.0 + 0.0 * t], dtype=float),
        velocity=lambda t: np.array([2.0 + 0.0 * t, 0.0 * t, -1.0 + 0.0 * t], dtype=float),
    )

    np.testing.assert_allclose(np.array(trajectory.location(0.5)), np.array([0.5, 0.0, 1.0]))
    np.testing.assert_allclose(np.array(trajectory.velocity(0.5)), np.array([2.0, 0.0, -1.0]))


def test_trajectory_accepts_vectorized_3d_outputs():
    """``Trajectory`` should accept vectorized 3D outputs with shape ``(3, N)``."""
    trajectory = ss.Trajectory(
        location=lambda t: np.array([t, t**2, 1.0 + 0.0 * t], dtype=float),
        velocity=lambda t: np.array([1.0 + 0.0 * t, 2.0 * t, 0.0 * t], dtype=float),
    )

    times = np.array([0.0, 1.0, 2.0])
    location = trajectory.location(times)
    velocity = trajectory.velocity(times)

    np.testing.assert_allclose(np.array(location), np.array([[0.0, 1.0, 2.0], [0.0, 1.0, 4.0], [1.0, 1.0, 1.0]]))
    np.testing.assert_allclose(np.array(velocity), np.array([[1.0, 1.0, 1.0], [0.0, 2.0, 4.0], [0.0, 0.0, 0.0]]))


def test_trajectory_accepts_integer_time_inputs():
    """``Trajectory`` should accept integer time inputs."""
    trajectory = ss.Trajectory(
        location=lambda t: np.array([t, 0.0 * t, 1.0 + 0.0 * t], dtype=float),
        velocity=lambda t: np.array([1.0 + 0.0 * t, 0.0 * t, 0.0 * t], dtype=float),
    )

    np.testing.assert_allclose(np.array(trajectory.location(1)), np.array([1.0, 0.0, 1.0]))
    np.testing.assert_allclose(np.array(trajectory.velocity(1)), np.array([1.0, 0.0, 0.0]))


def test_trajectory_validates_location_shape_on_assignment():
    """``Trajectory`` should reject location callables with invalid output shape."""
    with pytest.raises(ValueError, match='Trajectory output must have shape'):
        ss.Trajectory(
            location=lambda t: np.asarray([0.0, 1.0]) + 0 * np.asarray(t),
            velocity=lambda t: np.asarray([1.0, 0.0, 0.0]) + 0 * np.asarray(t),
        )


def test_spline_trajectory_location_matches_sampled_points():
    """``SplineTrajectory.location`` should pass through the sampled points."""
    trajectory = ss.SplineTrajectory(
        times=[0.0, 1.0, 2.0],
        locations=[[0.0, 0.0, 0.0], [1.0, 2.0, 0.0], [2.0, 4.0, 0.0]],
    )

    location = np.array(trajectory.location(1.0))

    np.testing.assert_allclose(location, np.array([1.0, 2.0, 0.0]))


def test_spline_trajectory_velocity_matches_linear_slope():
    """``SplineTrajectory.velocity`` should return the spline derivative."""
    trajectory = ss.SplineTrajectory(times=[0.0, 1.0], locations=[[0.0, 0.0, 0.0], [2.0, 1.0, -1.0]])

    velocity = np.array(trajectory.velocity(0.5))

    np.testing.assert_allclose(velocity, np.array([2.0, 1.0, -1.0]))


def test_spline_trajectory_accepts_integer_inputs():
    """``SplineTrajectory`` should accept integer sample data and query times."""
    trajectory = ss.SplineTrajectory(times=[0, 1], locations=[[0, 0, 0], [2, 0, 0]])

    np.testing.assert_allclose(np.array(trajectory.location(1)), np.array([2.0, 0.0, 0.0]))
    np.testing.assert_allclose(np.array(trajectory.velocity(1)), np.array([2.0, 0.0, 0.0]))


def test_spline_trajectory_supports_vectorized_queries():
    """``SplineTrajectory`` should support array-valued query times."""
    trajectory = ss.SplineTrajectory(times=[0.0, 1.0], locations=[[0.0, 0.0, 0.0], [2.0, 1.0, -1.0]])

    times = np.array([0.0, 0.5, 1.0])
    location = np.array(trajectory.location(times))
    velocity = np.array(trajectory.velocity(times))

    np.testing.assert_allclose(location, np.array([[0.0, 1.0, 2.0], [0.0, 0.5, 1.0], [0.0, -0.5, -1.0]]))
    np.testing.assert_allclose(velocity, np.array([[2.0, 2.0, 2.0], [1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]]))


def test_spline_trajectory_requires_at_least_two_times():
    """``SplineTrajectory`` should reject underspecified splines."""
    trajectory = ss.SplineTrajectory(times=[0.0], locations=[[0.0, 0.0, 0.0]])

    with pytest.raises(ValueError, match='at least two samples'):
        trajectory.location(0.0)


def test_spline_trajectory_requires_matching_location_shape():
    """``SplineTrajectory`` should validate the shape of the sampled locations."""
    trajectory = ss.SplineTrajectory(times=[0.0, 1.0], locations=[[0.0, 0.0], [1.0, 0.0]])

    with pytest.raises(ValueError, match='locations must have shape'):
        trajectory.location(0.0)


def test_spline_trajectory_requires_one_dimensional_times():
    """``SplineTrajectory`` should reject non-one-dimensional time arrays."""
    trajectory = ss.SplineTrajectory(times=[[0.0, 1.0]], locations=[[0.0, 0.0, 0.0]])

    with pytest.raises(ValueError, match='times must be a one-dimensional array'):
        trajectory.location(0.0)


def test_spline_trajectory_requires_two_distinct_times():
    """``SplineTrajectory`` should reject sample times that collapse to one instant."""
    trajectory = ss.SplineTrajectory(times=[0.0, 0.0], locations=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    with pytest.raises(ValueError, match='times must contain at least two distinct samples'):
        trajectory.location(0.0)


def test_spline_trajectory_sorts_and_deduplicates_times():
    """``SplineTrajectory`` should accept unsorted or repeated times when consistent."""
    trajectory = ss.SplineTrajectory(
        times=[1.0, 0.0, 0.0, 2.0],
        locations=[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
    )

    np.testing.assert_allclose(trajectory.prepared_times, np.array([0.0, 1.0, 2.0]))
    np.testing.assert_allclose(np.array(trajectory.location(1.0)), np.array([1.0, 0.0, 0.0]))


def test_spline_trajectory_rejects_conflicting_duplicate_times():
    """``SplineTrajectory`` should reject duplicate times with conflicting locations."""
    trajectory = ss.SplineTrajectory(times=[0.0, 0.0], locations=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    with pytest.raises(ValueError, match='duplicate times must map to identical locations'):
        trajectory.location(0.0)
