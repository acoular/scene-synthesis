import numpy as np
from acoular import SignalGenerator, Trajectory
from traits.api import CArray, HasStrictTraits, Instance

from scene_synthesis.directivities import Directivity


class Source(HasStrictTraits):
    """Class representing an acoustic source.

    Examples
    --------
    Instantiate a simple source with a sine signal and a default trajectory:

    >>> from acoular import SineGenerator, Trajectory
    >>> from scene_synthesis.sources import Source
    >>> signal = SineGenerator(frequency=1000, sample_freq=44100, num_samples=44100)
    >>> trajectory = Trajectory()
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
