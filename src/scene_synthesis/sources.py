from acoular import SignalGenerator

from traits.api import CArray, Instance, HasStrictTraits


class Source(HasStrictTraits):
    #: The signal of the source.
    signal = Instance(SignalGenerator)

    #: The (initial) 3D location of the source.
    location = CArray(shape=(3,), dtype=float)
