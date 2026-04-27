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
    The return values are normalized to the component-wise convention used by
    the rest of scene-synthesis: ``[x, y, z]`` for scalar times and three
    arrays ``[x(t), y(t), z(t)]`` for array-valued times.

    Examples
    --------
    >>> import scene_synthesis as ss
    >>> location = lambda t: np.stack(
    ...     [np.asarray(t), np.zeros_like(t), np.ones_like(t)],
    ...     axis=-1,
    ... )
    >>> velocity = lambda t: np.stack(
    ...     [np.ones_like(t), np.zeros_like(t), np.zeros_like(t)],
    ...     axis=-1,
    ... )
    >>> traj = ss.Trajectory(location=location, velocity=velocity)
    >>> traj.location(0.5)
    [array(0.5), array(0.), array(1.)]
    >>> traj.velocity(0.5)
    [array(1.), array(0.), array(0.)]
    """

    #: Time-dependent position function.
    location = Property(desc='time-dependent position function')

    #: Backing trait for :attr:`location`.
    _location = Callable

    #: Time-dependent velocity function.
    velocity = Property(desc='time-dependent velocity function')

    #: Backing trait for :attr:`velocity`.
    _velocity = Callable

    @staticmethod
    def _normalize_output(value):
        """Normalize trajectory outputs to three component arrays."""
        if isinstance(value, (list, tuple)) and len(value) == 3:
            return [np.asarray(component, dtype=float) for component in value]

        array = np.asarray(value, dtype=float)
        if array.shape == (3,):
            return [np.asarray(array[0]), np.asarray(array[1]), np.asarray(array[2])]
        if array.ndim >= 2 and array.shape[-1] == 3:
            return [np.asarray(array[..., 0]), np.asarray(array[..., 1]), np.asarray(array[..., 2])]
        if array.ndim >= 1 and array.shape[0] == 3:
            return [np.asarray(array[0, ...]), np.asarray(array[1, ...]), np.asarray(array[2, ...])]

        msg = f'Trajectory output must describe 3D coordinates, got shape {array.shape}.'
        raise ValueError(msg)

    def _get_location(self):
        return lambda t: self._normalize_output(self._location(t))

    def _set_location(self, value):
        if not callable(value):
            msg = 'location must be callable.'
            raise ValueError(msg)
        self._location = value

    def _get_velocity(self):
        return lambda t: self._normalize_output(self._velocity(t))

    def _set_velocity(self, value):
        if not callable(value):
            msg = 'velocity must be callable.'
            raise ValueError(msg)
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
    [array(0.5), array(0.), array(0.)]
    >>> trajectory.velocity(0.5)
    [array(1.), array(0.), array(0.)]
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

        super().__init__(location=self._location_spline.__call__, velocity=self._velocity_spline.__call__)

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
        """Sort sample times and merge duplicate times with identical locations."""
        order = np.argsort(self.times)
        sorted_times = self.times[order]
        sorted_locations = self.locations[order]

        unique_times = [sorted_times[0]]
        unique_locations = [sorted_locations[0]]
        for time, location in zip(sorted_times[1:], sorted_locations[1:], strict=True):
            if np.isclose(time, unique_times[-1]):
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
