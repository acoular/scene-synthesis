"""Acoustic source definition and properties."""

import numpy as np
from acoular import SignalGenerator
from traits.api import CArray, HasStrictTraits, Instance

from scene_synthesis.directivities import Directivity
from scene_synthesis.trajectory import Trajectory


class Source(HasStrictTraits):
    """Class representing an acoustic source.

    Examples
    --------
    Instantiate a simple source with a sine signal and a fixed trajectory:

    >>> from acoular import SineGenerator
    >>> from scene_synthesis import FixedTrajectory, Source
    >>> signal = SineGenerator(freq=1000, sample_freq=44100, num_samples=44100)
    >>> trajectory = FixedTrajectory(points={0.0: (0.0, 0.0, 0.0), 1.0: (1.0, 0.0, 0.0)})
    >>> source = Source(signal=signal, trajectory=trajectory)
    """

    #: The signal of the source.
    signal = Instance(SignalGenerator)

    #: The (initial) 3D location of the source.
    location = CArray(shape=(3,), dtype=float)

    #: The trajectory of the source.
    trajectory = Instance(Trajectory)

    #: Whether to apply amplitude convection correction.
    conv_amp = Instance(bool, value=False)

    #: Vectors defining the (initial) global orientation of the source
    #: These vectors must be orthogonal to each other
    #: self.orientation[0] = right_vec
    #: self.orientation[1] = up_vec
    #: self.orientation[2] = forward_vec
    orientation = CArray(shape=(3, 3), desc='source orientation matrix', value=np.eye(3))

    #: The directivity of the source.
    directivity = Instance(Directivity)
