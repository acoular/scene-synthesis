"""Test cases for acoustic sources."""

import acoular as ac
import scene_synthesis as ss
from pytest_cases import parametrize_with_cases

from tests.cases_trajectory import Trajectories


class Sources:
    """Test cases for acoustic sources."""

    @parametrize_with_cases('trajectory', cases=Trajectories)
    def case_single(self, trajectory):
        """Single source test case."""
        n = 10000
        signal = ac.SineGenerator(freq=0.1, num_samples=n, sample_freq=n)
        return [ss.Source(signal=signal, trajectory=trajectory, location=[1, 0, 0])]
