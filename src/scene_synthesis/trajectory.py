"""Trajectory definitions for scene synthesis.

This module separates the abstract trajectory interface from the current
fixed-frame implementation. That keeps the existing point-trajectory use case
available while leaving room for future motion models with moving local
reference frames.
"""

import numpy as np
from scipy.interpolate import splev, splprep
from traits.api import Dict, Float, HasStrictTraits, Property, Tuple, cached_property, property_depends_on


class Trajectory(HasStrictTraits):
    """Abstract trajectory interface.

    A trajectory maps time to a 3D position and optionally provides time
    derivatives such as velocity. Concrete implementations may represent only a
    point moving in the global frame or, in the future, trajectories derived
    from richer motion/reference-frame descriptions.
    """

    def location(self, t, der=0):
        """Evaluate the trajectory or one of its derivatives at time ``t``."""
        raise NotImplementedError

    def shift_by_offset(self, x_off):
        """Return a shifted copy of the trajectory."""
        raise NotImplementedError


class FixedTrajectory(Trajectory):
    """Represent a point trajectory in a fixed frame of reference.

    The trajectory is specified by a mapping from time instants to sampled
    ``(x, y, z)`` positions in the global frame. A spline is fit through those
    samples and can then be evaluated at arbitrary times to obtain positions or
    time derivatives such as velocity.

    Notes
    -----
    - The spline order is chosen automatically based on the number of
      available points, up to cubic interpolation.
    - The frame of reference is fixed, i.e. the sampled positions are
      interpreted directly as global coordinates.

    Examples
    --------
    >>> import scene_synthesis as ss
    >>> trajectory = ss.FixedTrajectory(points={0.0: (0.0, 0.0, 0.0), 1.0: (1.0, 0.0, 0.0)})
    >>> trajectory.location(0.5)
    [array(0.5), array(0.), array(0.)]
    """

    #: Dictionary mapping time instants to sampled ``(x, y, z)`` positions.
    points = Dict(
        key_trait=Float,
        value_trait=Tuple(Float, Float, Float),
    )

    #: Start and end time of the trajectory as ``(t_min, t_max)``.
    interval = Property()

    #: Internal spline representation returned by :func:`scipy.interpolate.splprep`.
    tck = Property()

    @property_depends_on(['points[]'])
    def _get_interval(self):
        if not self.points:
            msg = 'Trajectory.points must contain at least one sampled position to compute an interval.'
            raise ValueError(msg)
        return np.sort(list(self.points.keys()))[np.r_[0, -1]]

    @cached_property
    @property_depends_on(['points[]'])
    def _get_tck(self):
        if len(self.points) < 2:
            msg = 'Trajectory.points must contain at least two sampled positions to build a spline.'
            raise ValueError(msg)
        t = np.sort(list(self.points.keys()))
        xp = np.array([self.points[i] for i in t]).T
        k = min(3, len(self.points) - 1)
        tcku = splprep(xp, u=t, s=0, k=k)
        return tcku[0]

    def location(self, t, der=0):
        """Evaluate the trajectory or one of its derivatives.

        Parameters
        ----------
        t : float or array-like of float
            Time instant or time instants at which to evaluate the trajectory.
        der : int, optional
            Derivative order. Use ``0`` for position, ``1`` for velocity,
            ``2`` for acceleration, and so on. Defaults to ``0``.

        Returns
        -------
        list[numpy.ndarray]
            Three arrays representing the ``x``, ``y``, and ``z`` components
            at the requested times.

        Examples
        --------
        >>> import scene_synthesis as ss
        >>> trajectory = ss.FixedTrajectory(points={0.0: (0.0, 0.0, 0.0), 1.0: (1.0, 0.0, 0.0)})
        >>> trajectory.location(0.5)
        [array(0.5), array(0.), array(0.)]
        """
        return splev(t, self.tck, der)

    def shift_by_offset(self, x_off):
        """Return a copy of the trajectory shifted by a constant 3D offset.

        Parameters
        ----------
        x_off : array-like of float
            Offset added to every sampled point.

        Returns
        -------
        FixedTrajectory
            Shifted trajectory with the same time samples.
        """
        offset = np.asarray(x_off, dtype=float)
        shifted_points = {time: tuple(np.asarray(point, dtype=float) + offset) for time, point in self.points.items()}
        return FixedTrajectory(points=shifted_points)

    def traj(self, t_start, t_end=None, delta_t=None, der=0):
        """Iterate through trajectory samples over a time range.

        Parameters
        ----------
        t_start : float
            Start time of the iteration. If ``delta_t`` is omitted, this value
            is interpreted as the step size and the full trajectory interval is
            used.
        t_end : float, optional
            End time of the iteration. Defaults to the end of
            :attr:`interval`.
        delta_t : float, optional
            Time step between yielded samples. If omitted, ``t_start`` is used
            as the step size for traversing the full trajectory interval.
        der : int, optional
            Derivative order to evaluate. Defaults to ``0``.

        Yields
        ------
        tuple[numpy.float64, numpy.float64, numpy.float64]
            The interpolated ``(x, y, z)`` values at each sampled time.

        Examples
        --------
        >>> import scene_synthesis as ss
        >>> trajectory = ss.FixedTrajectory(points={0.0: (0.0, 0.0, 0.0), 1.0: (1.0, 0.0, 0.0)})
        >>> samples = list(trajectory.traj(0.5))
        >>> samples[0]
        (np.float64(0.0), np.float64(0.0), np.float64(0.0))
        >>> samples[1]
        (np.float64(0.5), np.float64(0.0), np.float64(0.0))
        >>> samples = list(trajectory.traj(0.0, 1.0, 0.5))
        >>> samples[0]
        (np.float64(0.0), np.float64(0.0), np.float64(0.0))
        >>> samples[1]
        (np.float64(0.5), np.float64(0.0), np.float64(0.0))
        """
        if delta_t is None:
            delta_t = t_start
            t_start, t_end = self.interval
        if delta_t <= 0:
            msg = 'delta_t must be a positive time step.'
            raise ValueError(msg)
        if t_end is None:
            t_end = self.interval[1]
        yield from zip(*self.location(np.arange(t_start, t_end, delta_t), der), strict=True)
