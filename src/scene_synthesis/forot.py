"""Frame of Reference over Time."""
from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.transform import RigidTransform
from scipy.spatial.transform import Rotation as ScipyRotation

from .trajectory import CircularTrajectory, Trajectory


class FOROT(ABC):
    """Representation of a Local Reference Frame.

    Typically this would be the body-fixed reference frame of the moving
    source region (e.g. aircraft).
    """

    @abstractmethod
    def trajectory(self, x_local: ArrayLike) -> Trajectory: ...

    @abstractmethod
    def rigid_transform(self, t: float) -> RigidTransform: ...


class Translation(FOROT):
    """Local Reference Frame where the Orientation is constant.

    Parameters
    ----------
    origin_traj: Trajectory
        The trajectory of the origin of the local reference frame.
    orientation: np.ndarray
        'xyz'-Euler angles in degrees of a rotation that is applied to the
        reference frame after the shift due to translation.
    """

    def __init__(
        self,
        origin_traj: Trajectory,
        orientation_angles: ArrayLike = (0.0, 0.0, 0.0),
    ):
        self.origin_traj = origin_traj
        self.orientation = ScipyRotation.from_euler(
            'xyz', orientation_angles, degrees=True
        )

    def trajectory(self, x_local: ArrayLike) -> Trajectory:
        shift = self.orientation.apply(x_local)
        return self.origin_traj.shift_by_offset(shift)

    def rigid_transform(self, t: float) -> RigidTransform:
        return RigidTransform.from_components(self.origin_traj.location(t), self.orientation)

    def partial_time_derivative(self, x_local: ArrayLike, t: float) -> NDArray:
        # Derivative is independent of location in space.
        return self.origin_traj.location(t, der=1)


class Rotation(FOROT):
    """Local Reference Frame that rotates around some axis at a constant rate.

    Parameters
    ----------
    rev_per_sec: float
        Rate of rotation (in revolutions per second).
    orientation: np.ndarray, optional
        'xyz'-Euler angles in degrees of a rotation that is applied to the
        reference frame after the shift due to translation.
        This rotation is applied after the rotation around `axis` but before the
        shift due to `origin`
    axis: {x, y, z}, optional
        The axis around which to rotate.
    """

    base = {
        'x': np.array([1., 0., 0.]),
        'y': np.array([0., 1., 0.]),
        'z': np.array([0., 0., 1.])
    }

    def __init__(
        self,
        rev_per_sec: float=1.,
        orientation_angles: ArrayLike=(0.,0.,0.),
        origin: ArrayLike=(0.,0.,0.),
        axis: str='z'
    ):
        self.origin = np.asarray(origin)
        self.const_rotation = ScipyRotation.from_euler(
            'xyz', orientation_angles, degrees=True
        )
        self.rev_per_sec = rev_per_sec
        self.axis = self.base[axis]

    def trajectory(self, x_local: ArrayLike) -> CircularTrajectory:
        return CircularTrajectory(
            self.rev_per_sec,
            self.origin + self.const_rotation.apply(x_local),
            self.const_rotation.apply(self.axis),
            self.origin
        )

    def rigid_transform(self, t: float) -> RigidTransform:
        return RigidTransform.from_components(
            self.origin,
            self.const_rotation * ScipyRotation.from_rotvec(t * self.rev_per_sec * 2 * np.pi * self.axis)
        )

    def partial_time_derivative(self, x_local: ArrayLike, t: float) -> NDArray:
        # Derivative is 90° rotation from orignal location, without any translation
        # Note: Untested!
        omega = self.rev_per_sec * 2 * np.pi

        rot = ScipyRotation.from_rotvec(t * omega + np.pi / 2 * self.axis)

        return omega * self.const_rotation.apply(rot.apply(x_local))
