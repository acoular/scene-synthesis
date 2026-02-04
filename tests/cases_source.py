import acoular as ac
import scene_synthesis as ss
from pytest_cases import parametrize_with_cases

from tests.cases_trajectory import Trajectories


class Sources:
    @parametrize_with_cases('trajectory', cases=Trajectories)
    def case_single(self, trajectory):
        N = 10000
        signal = ac.SineGenerator(freq=0.1, num_samples=N, sample_freq=N)
        return [ss.Source(signal=signal, trajectory=trajectory)]
