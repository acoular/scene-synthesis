from acoular import SamplesGenerator, Environment, SignalGenerator
from traits.api import CList, Instance, HasStrictTraits, Float, Tuple
import numpy as np


class Sources(HasStrictTraits):
    signals = CList(Instance(SignalGenerator))

    loc = CArray(shape=(3,), default=(0,0,0))

    start_t = Float(0.0)

    start = Float(0.0)

