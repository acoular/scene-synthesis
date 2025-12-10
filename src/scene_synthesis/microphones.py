from traits.api import CArray, HasStrictTraits


class Microphone(HasStrictTraits):
    #: The 3D location of the microphone.
    location = CArray(shape=(3,), dtype=float)
