"""Acoustic source definition and properties."""

import numpy as np
from acoular import SignalGenerator, Trajectory
from traits.api import Bool, CArray, HasStrictTraits, Instance

from scene_synthesis.directivities import Directivity


class Source(HasStrictTraits):
    """Class representing an acoustic source.

    Examples
    --------
    Instantiate a simple source with a sine signal and a default trajectory:

    >>> from acoular import SineGenerator, Trajectory
    >>> from scene_synthesis.sources import Source
    >>> signal = SineGenerator(freq=1000, sample_freq=44100, num_samples=44100)
    >>> trajectory = Trajectory()
    >>> source = Source(signal=signal, trajectory=trajectory)
    """

    #: Virtual source-time signal.
    #:
    #: Samples live on the source/emission time axis. The base class does not
    #: define the physical quantity represented by this signal. Subclasses must
    #: specify whether samples represent pressure, volume velocity, force,
    #: acceleration, or another source-strength quantity.
    signal = Instance(SignalGenerator)

    #: The (initial) 3D location of the source.
    location = CArray(shape=(3,), dtype=float)

    #: The trajectory of the source.
    trajectory = Instance(Trajectory)

    #: Whether to apply amplitude convection correction.
    conv_amp = Bool(False)

    #: Vectors defining the (initial) global orientation of the source
    #: These vectors must be orthogonal to each other
    #: self.orientation[0] = right_vec
    #: self.orientation[1] = up_vec
    #: self.orientation[2] = forward_vec
    orientation = CArray(shape=(3, 3), desc='source orientation matrix', value=np.eye(3))

    #: The directivity of the source.
    directivity = Instance(Directivity)

    def local_strength(self, environment):  # noqa: ARG002
        """Return source-time local strength signal.

        Base source keeps legacy behavior: the signal is already the local
        strength. Physical source subclasses should override this method so
        frequency-dependent radiation laws are always applied before
        propagation.
        """
        return np.asarray(self.signal.signal(), dtype=float)

    def direction_factor(self, target_directions):
        """Return direction-dependent source factor.

        Base source is omnidirectional. ``target_directions`` has shape
        ``(3, N)`` and contains source-to-receiver unit vectors.
        """
        return np.ones(target_directions.shape[1], dtype=float)


class MonopoleSource(Source):
    """Ideal monopole source driven by volume velocity.

    For a monopole, :attr:`signal` is assumed to be given as the source volume
    velocity signal ``Q(t)`` in ``m**3/s``.

    The free-field far-field pressure of a stationary monopole in time domain is

    ``p(r, t) = rho0 / (4*pi*r) * dQ/dt(t - r/c)``.

    Since propagation delay and geometric spreading are handled by the scene,
    the source provides a local source strength which handles the radiation law
    of the source. The time reference is source-based, meaning the pressure
    strength colocated with the source is reduced to

    ``p_s(t) = rho0 / (4*pi) * dQ/dt(t)``.
    """

    def local_strength(self, environment):
        q = self.signal.signal()
        return environment.rho0 / (4 * np.pi) * np.diff(q, prepend=0.0) * self.signal.sample_freq


class DipoleSource(Source):
    """Ideal compact dipole source driven by prescribed force.

    For a dipole, :attr:`signal` is assumed to be the source-time point force
    ``F_p(t)`` in ``N`` acting on the fluid along ``orientation[2]``. This is
    analogous to :class:`MonopoleSource`, where the signal prescribes volume
    velocity ``Q(t)`` directly: radiation impedance matters for source loading,
    radiated power, and mapping an actuator drive to ``F_p(t)``, but this ideal
    source model does not solve that coupling.

    The source center is :attr:`location`. The dipole axis is
    ``orientation[2]``. The time-domain equation has the same
    propagation shape as the monopole equation:

    ``p(r, t) = cos(theta) / r * dF_p/dt(t - r/c) / (4*pi*c)``.

    The factor ``cos(theta)`` is geometric and must be computed from the dipole
    axis and the source-to-receiver direction during propagation.

    Separation theorem for this compact model: a finite opposite-monopole pair
    has coupled frequency/direction response ``2j*sin(k*d*cos(theta)/2)``. In
    the point-dipole limit, with finite spacing condensed into ``F_p(t)``, this
    becomes the separable form ``j*k * F_p * cos(theta)``. The ``j*k`` high-pass
    is represented here by the time derivative in :meth:`force_derivative`;
    :meth:`direction_factor` keeps only the compact-limit angular term. Finite
    spacing effects such as lobing require a finite-pair model instead.
    """

    def axis(self):
        """Return normalized global dipole axis."""
        axis = np.asarray(self.orientation[2], dtype=float)
        norm = np.linalg.norm(axis)
        if norm == 0.0:
            msg = 'dipole axis must be non-zero'
            raise ValueError(msg)
        return axis / norm

    def force_derivative(self):
        """Return source-time ``dF/dt``."""
        return np.diff(self.signal.signal(), prepend=0.0) * self.signal.sample_freq

    def direction_factor(self, target_directions):
        """Return ``cos(theta)`` for source-to-receiver directions.

        ``target_directions`` must have shape ``(3, N)``. Each column is a unit
        vector from source to receiver in global coordinates.
        """
        return np.dot(self.axis(), target_directions)

    def local_strength(self, environment):
        """Return source-time force-dipole strength.

        Propagation must apply retarded time, geometric spreading ``1/r``, and
        ``cos(theta)`` from :meth:`direction_factor`.
        """
        return self.force_derivative() / (4 * np.pi * environment.c)
