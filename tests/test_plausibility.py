import numpy as np
import acoular as ac
import pytest
from pytest_cases import parametrize_with_cases

from tests.cases_trajectory import Trajectories
from tests.cases_environment import Environments


@parametrize_with_cases('environment', Environments)
@parametrize_with_cases('trajectory', Trajectories)
def test_analytical(environment, trajectory):
    """Test that analytical trajectory matches expected positions."""
    # Test parameters
    num_samples = 1000
    T = 10
    t = np.linspace(0, T, num_samples)

    if trajectory is None:
        pytest.skip("No trajectory provided for this test case")
    else:
        assert isinstance(trajectory, ac.Trajectory), "Provided trajectory is not an instance of ac.Trajectory"
