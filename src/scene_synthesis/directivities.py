"""Directivity models for acoustic sources and microphones."""

import numpy as np
from traits.api import ABCHasStrictTraits


class Directivity(ABCHasStrictTraits):
    """Base class for directivity patterns."""

    def get_coefficients(self, orientation, target_directions):  # noqa: ARG002
        """Get directivity coefficients."""
        msg = 'Directivity.get_coefficients must be implemented by subclasses'
        raise NotImplementedError(msg)


class OmniDirectivity(Directivity):
    """Omnidirectional directivity pattern."""

    def get_coefficients(self, orientation, target_directions):  # noqa: ARG002
        """Get omnidirectional coefficients."""
        return np.ones(target_directions.shape[1], dtype=float)


class CardioidDirectivity(Directivity):
    """Cardioid directivity pattern."""

    def get_coefficients(self, orientation, target_directions):  # noqa: ARG002
        """Get cardioid coefficients."""
        return 0.5 * (1.0 + np.dot(orientation[2], target_directions))
