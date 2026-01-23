import numpy as np
from acoular import SignalGenerator, Trajectory
from traits.api import CArray, HasStrictTraits, Instance

from scene_synthesis.directivities import Directivity


class Source(HasStrictTraits):
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
