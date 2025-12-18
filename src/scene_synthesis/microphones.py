from scene_synthesis import Directivity
from traits.api import CArray, HasStrictTraits, Instance


class Microphone(HasStrictTraits):
    #: The 3D location of the microphone.
    location = CArray(shape=(3,), dtype=float)

    #: The directivity of the microphone
    directivity = Instance(Directivity)
