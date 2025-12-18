from traits.api import CArray, ABCHasStrictTraits, Property
import numpy as np


class Directivity(ABCHasStrictTraits):

    def get_coefficients(self, orientation, target_directions):
        pass


class OmniDirectivity(Directivity):
    def get_coefficients(self, orientation, target_directions):
        return np.ones(target_directions.shape[1], dtype=float)


class CardioidDirectivity(Directivity):
    def get_coefficients(self, orientation, target_directions):
        return 0.5 * (1.0 + np.dot(orientation[2], target_directions))
