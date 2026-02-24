"""Test cases for microphone configurations."""

import numpy as np
import scene_synthesis as ss


class Microphones:
    """Test cases for microphone configurations."""

    def case_single(self):
        """Single microphone test case."""
        return [ss.Microphone(location=np.array((0, 0, 0)))]

    def case_array(self):
        """Microphone array test case."""
        return [ss.Microphone(location=np.array((x, y, 0))) for x, y in ((-0.5, 0), (0, 0), (0, 0.5))]
