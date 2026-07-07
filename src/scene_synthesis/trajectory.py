"""Frame of Reference over Time."""
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Self

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import make_interp_spline


class Trajectory(ABC):
    """Representation of a Trajectory."""

    @abstractmethod
    def location(self, t: ArrayLike) -> NDArray:
        """Location over time.

        Result hase shape ``(3,)`` for scalar values times and ``(3,N)`` for
        array-valued times.
        """
        ...

    @abstractmethod
    def velocity(self, t: ArrayLike) -> NDArray:
        """Velocity over time.

        Implementing Base Class must guarantee correctness.
        """
        ...

    @abstractmethod
    def shift_by_offset(self, x_off: ArrayLike) -> Self:
        """Return a copy of this trajectory shifted in space by some const. 3D-offset."""
        ...

class UniformLinearTrajectory(Trajectory):
    """Motion with constant speed."""

    def __init__(self, x0: ArrayLike, v0: ArrayLike):
        """
        Parameters
        ----------
        x0 : ArrayLike of shape (3,)
            Position at t=0.
        v0 : ArrayLike of shape (3,)
            Velocity.
        """  # noqa: D205
        self.x0 = np.asarray(x0)
        self.v0 = np.asarray(v0)

    def location(self, t):
        t = np.asarray(t)
        if t.ndim == 0:
            return self.x0 + t * self.v0
        return self.x0[:, np.newaxis] + t.flatten() * self.v0[:, np.newaxis]

    def velocity(self, t):
        t = np.asarray(t)
        if t.ndim == 0:
            return self.v0
        return np.broadcast_to(self.v0, t.size + (3,)).T

    def shift_by_offset(self, x_off):
        return UniformLinearTrajectory(self.x0 + x_off, self.v0)


class SplineTrajectory(Trajectory):
    """Motion defined by some spline interpolation."""

    def __init__(self, times, positions, *args, **kwargs):
        self._bspline_obj = make_interp_spline(times, positions, *args, **kwargs)

    def location(self, t):
        return self._bspline_obj(t, nu=0).T

    def velocity(self, t):
        return self._bspline_obj(t, nu=1).T


    def shift_by_offset(self, x_off):
        cp = deepcopy(self)
        cp._bspline_obj.c += x_off
        return cp


class CircularTrajectory(Trajectory):
    """Represents a circular motion of a point around some center.

    Does so by storing a centerpoint `_center_point`, a vector from the center
    point to the initial point `_xc` and a vector perpendicular to both the
    axis of rotation and `_xc`.

    The location is calculated as the sum of the center point, the cosine-weighted
    `_xc` and the sine-weighted `_xs`.
    """

    def __init__(
        self,
        rev_per_sec: float,
        ref_pos: ArrayLike,
        axis_dir: ArrayLike = (0.0, 0.0, 1.0),
        fixpoint: ArrayLike = (0.0, 0.0, 0.0),
    ):
        """
        Parameters
        ----------
        ref_pos : ArrayLike of shape (3,)
            Position at t=0
        rev_per_sec : float
            Speed of rotation in revolutions per second
        axis_dir : ArrayLike of shape (3,), optional
            The direction of the axis of rotation.
            Will be normalized. Must have length > 0.
        fixpoint : ArrayLike of shape (3,), optional
            A point on the axis of rotation.
        """  # noqa: D205
        self.rev_per_sec = rev_per_sec

        ref_pos = np.asarray(ref_pos)
        fixpoint = np.asarray(fixpoint)

        axis_dir = np.asarray(axis_dir) / np.linalg.norm(axis_dir)
        dx = ref_pos - fixpoint

        self._center_point = fixpoint + axis_dir * np.dot(axis_dir, dx)
        self._xc = ref_pos - self._center_point
        self._xs = np.cross(axis_dir, self._xc)

    def location(self, t):
        t = np.asarray(t)
        omega = self.rev_per_sec * 2 * np.pi
        if t.ndim == 0:
            return self._center_point + self._xc * np.cos(omega * t) + self._xs * np.sin(omega * t)
        t = t.flatten()
        return (
            self._center_point[:, np.newaxis]
             + self._xc[:, np.newaxis] * np.cos(omega * t)
             + self._xs[:, np.newaxis] * np.sin(omega * t)
        )

    def velocity(self, t):
        t = np.asarray(t)
        omega = self.rev_per_sec * 2 * np.pi
        if t.ndim == 0:
            return omega * ( - self._xc * np.sin(omega * t) + self._xs * np.cos(omega * t))
        t = t.flatten()
        return omega * (
             - self._xc[:, np.newaxis] * np.sin(omega * t)
             + self._xs[:, np.newaxis] * np.cos(omega * t)
        )

    def shift_by_offset(self, x_off):
        cp = deepcopy(self)
        cp._center_point += x_off
        return cp
