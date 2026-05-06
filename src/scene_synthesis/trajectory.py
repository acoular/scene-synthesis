"""Trajectory definitions for scene synthesis."""

import numpy as np
from scipy.interpolate import make_interp_spline
from traits.api import Array, Callable, HasStrictTraits, Property, cached_property


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
        self._validate_output_shape(value(np.array([0.0, 1.0])))
        self._location = value

    def _get_velocity(self):
        return self._velocity

    def _set_velocity(self, value):
        self._validate_output_shape(value(0.0))
        self._validate_output_shape(value(np.array([0.0, 1.0])))
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

    #: Sorted and deduplicated sample times.
    prepared_times = Property(depends_on=['times', 'locations'])

    #: Sorted and deduplicated sample locations.
    prepared_locations = Property(depends_on=['times', 'locations'])

    #: Time-dependent position function.
    location = Property(depends_on=['times', 'locations'])

    #: Time-dependent velocity function.
    velocity = Property(depends_on=['times', 'locations'])

    #: Cached location spline.
    _location_spline = Property(depends_on=['times', 'locations'])

    #: Cached velocity spline.
    _velocity_spline = Property(depends_on=['times', 'locations'])

    def _prepared_samples(self):
        """Return sorted samples with exact duplicate times merged."""
        if self.times.ndim != 1:
            msg = f'times must be a one-dimensional array, got shape {self.times.shape}.'
            raise ValueError(msg)
        if self.times.size < 2:
            msg = 'times must contain at least two samples.'
            raise ValueError(msg)
        if self.locations.shape != (self.times.size, 3):
            msg = f'locations must have shape ({self.times.size}, 3), got {self.locations.shape}.'
            raise ValueError(msg)

        order = np.argsort(self.times)
        sorted_times = self.times[order]
        sorted_locations = self.locations[order]

        unique_times = [sorted_times[0]]
        unique_locations = [sorted_locations[0]]
        for time, location in zip(sorted_times[1:], sorted_locations[1:], strict=True):
            if time == unique_times[-1]:
                if not np.array_equal(location, unique_locations[-1]):
                    msg = 'duplicate times must map to identical locations.'
                    raise ValueError(msg)
                continue
            unique_times.append(time)
            unique_locations.append(location)

        prepared_times = np.asarray(unique_times, dtype=float)
        if prepared_times.size < 2:
            msg = 'times must contain at least two distinct samples.'
            raise ValueError(msg)

        prepared_locations = np.asarray(unique_locations, dtype=float)
        return prepared_times, prepared_locations

    def _get_prepared_times(self):
        return self._prepared_samples()[0]

    def _get_prepared_locations(self):
        return self._prepared_samples()[1]

    @cached_property
    def _get__location_spline(self):
        order = min(3, self.prepared_times.size - 1)
        return make_interp_spline(self.prepared_times, self.prepared_locations, k=order, axis=0)

    @cached_property
    def _get__velocity_spline(self):
        return self._location_spline.derivative()

    def _location_from_spline(self, t):
        values = np.asarray(self._location_spline(t), dtype=float)
        return values if values.shape == (3,) else values.T

    def _velocity_from_spline(self, t):
        values = np.asarray(self._velocity_spline(t), dtype=float)
        return values if values.shape == (3,) else values.T

    def _get_location(self):
        return self._location_from_spline

    def _get_velocity(self):
        return self._velocity_from_spline
