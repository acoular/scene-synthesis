import numpy as np
from traits.api import CArray, ABCHasStrictTraits, Property


class Directivity(ABCHasStrictTraits):
    #: The 3D orientation of the object.
    orientation = CArray(shape=(3, 3), desc='orientation matrix', value=np.eye(3))

    #: The direction of the target direction.
    target_directions = Property(desc='Directions of the other objects to which the directivity is calculated.')
    _target_directions = CArray(shape=(3, None), default=np.array([[0.0], [0.0], [1.0]]))

    #: Calculated directivity coefficients
    coefficients = Property(desc='Directivity coefficients', depends_on=['orientation', 'target_directions'])

    def _get_coefficients(self):
        pass


class OmniDirectivity(Directivity):
    def _get_coefficients(self):
        return np.ones(self.target_directions.shape[1], dtype=float)


class CardioidDirectivity(Directivity):
    def _get_coefficients(self):
        return 0.5 * (1.0 + np.dot(self.orientation[2], self.target_directions))