"""Trajectory definitions for scene synthesis."""

import numpy as np
from scipy.interpolate import make_interp_spline
from traits.api import Any, Array, Callable, HasStrictTraits, Property


class Trajectory(HasStrictTraits):
    """Trajectory with independently supplied location and velocity functions.

    Parameters
    ----------
    location : callable
        Function mapping time ``t`` to a 3D position.
    velocity : callable
        Function mapping time ``t`` to a 3D velocity. This does not need to be
        the time derivative of ``location``.

    Notes
    -----
    Trajectory callables must return arrays describing 3D coordinates. Scalar
    times must produce shape ``(3,)`` and array-valued times must produce shape
    ``(3, N)``.

    Examples
    --------
    >>> import scene_synthesis as ss
    >>> location = lambda t: np.array([t, 0.0 * t, 1.0 + 0.0 * t], dtype=float)
    >>> velocity = lambda t: np.array([1.0 + 0.0 * t, 0.0 * t, 0.0 * t], dtype=float)
    >>> traj = ss.Trajectory(location=location, velocity=velocity)
    >>> traj.location(0.5)
    array([0.5, 0. , 1. ])
    >>> traj.velocity(0.5)
    array([1., 0., 0.])
    """

    #: Time-dependent position function.
    location = Property()

    #: Backing trait for :attr:`location`.
    _location = Callable

    #: Time-dependent velocity function.
    velocity = Property()

    #: Backing trait for :attr:`velocity`.
    _velocity = Callable

    @staticmethod
    def _validate_output_shape(value):
        """Validate that trajectory outputs describe 3D coordinates."""
        array = np.asarray(value, dtype=float)
        if array.shape == (3,):
            return
        if array.ndim == 2 and array.shape[0] == 3:
            return

        msg = f'Trajectory output must have shape (3,) or (3, N), got {array.shape}.'
        raise ValueError(msg)

    def _get_location(self):
        return self._location

    def _set_location(self, value):
        self._validate_output_shape(value(0.0))
        self._location = value

    def _get_velocity(self):
        return self._velocity

    def _set_velocity(self, value):
        self._validate_output_shape(value(0.0))
        self._velocity = value


class SplineTrajectory(Trajectory):
    """Spline-based trajectory built from sampled times and locations.

    Parameters
    ----------
    times : array-like of float
        Sample times.
    locations : array-like of float
        Sample positions with shape ``(N, 3)`` matching ``times``.

    Notes
    -----
    The location spline order is chosen automatically up to cubic, which keeps
    the interpolated trajectory at least :math:`C^1` whenever the available
    number of samples permits it.

    Examples
    --------
    >>> import scene_synthesis as ss
    >>> trajectory = ss.SplineTrajectory(
    ...     times=[0.0, 1.0],
    ...     locations=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
    ... )
    >>> trajectory.location(0.5)
    array([0.5, 0. , 0. ])
    >>> trajectory.velocity(0.5)
    array([1., 0., 0.])
    """

    #: Sample times.
    times = Array(dtype=float)

    #: Sample locations with shape ``(N, 3)``.
    locations = Array(dtype=float)

    #: Internal spline objects.
    _location_spline = Any
    _velocity_spline = Any

    def __init__(self, times, locations):
        self.times = np.asarray(times, dtype=float)
        self.locations = np.asarray(locations, dtype=float)
        self._validate_inputs()
        self._prepare_samples()

        order = min(3, self.times.size - 1)
        self._location_spline = make_interp_spline(self.times, self.locations, k=order, axis=0)
        self._velocity_spline = self._location_spline.derivative()

        super().__init__(location=self._location_callable, velocity=self._velocity_callable)

    @staticmethod
    def _evaluate_spline(spline, t):
        """Return spline values using the trajectory output convention."""
        values = np.asarray(spline(t), dtype=float)
        if values.shape == (3,):
            return values
        return values.T

    def _location_callable(self, t):
        return self._evaluate_spline(self._location_spline, t)

    def _velocity_callable(self, t):
        return self._evaluate_spline(self._velocity_spline, t)

    def _validate_inputs(self):
        """Validate spline trajectory inputs."""
        if self.times.ndim != 1:
            msg = f'times must be a one-dimensional array, got shape {self.times.shape}.'
            raise ValueError(msg)
        if self.times.size < 2:
            msg = 'times must contain at least two samples.'
            raise ValueError(msg)
        if self.locations.shape != (self.times.size, 3):
            msg = f'locations must have shape ({self.times.size}, 3), got {self.locations.shape}.'
            raise ValueError(msg)

    def _prepare_samples(self):
        """Sort sample times and merge exact duplicate times with identical locations."""
        order = np.argsort(self.times)
        sorted_times = self.times[order]
        sorted_locations = self.locations[order]

        unique_times = [sorted_times[0]]
        unique_locations = [sorted_locations[0]]
        for time, location in zip(sorted_times[1:], sorted_locations[1:], strict=True):
            if time == unique_times[-1]:
                if not np.allclose(location, unique_locations[-1]):
                    msg = 'duplicate times must map to identical locations.'
                    raise ValueError(msg)
                continue
            unique_times.append(time)
            unique_locations.append(location)

        self.times = np.asarray(unique_times, dtype=float)
        self.locations = np.asarray(unique_locations, dtype=float)

        if self.times.size < 2:
            msg = 'times must contain at least two distinct samples.'
            raise ValueError(msg)
