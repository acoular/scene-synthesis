from scene_synthesis.directivities import Directivity
from traits.api import CArray, HasStrictTraits, Instance
import numpy as np


class Microphone(HasStrictTraits):
    #: The 3D location of the microphone.
    location = CArray(shape=(3,), dtype=float)

    #: Vectors defining the global orientation of the microphone
    #: These vectors must be orthogonal to each other
    #: self.orientation[0] = right_vec
    #: self.orientation[1] = up_vec
    #: self.orientation[2] = forward_vec
    orientation = CArray(shape=(3, 3), desc='source orientation matrix', value=np.eye(3))

    #: The directivity of the microphone
    directivity = Instance(Directivity)
