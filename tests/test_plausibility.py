import numpy as np
from scipy.optimize import fsolve


def test_analytical(scene):
    """Test that analytical solution matches synthesis result."""

    # analytical solution
    def arrival_time_equation(tau_0, tt):
        if scene.sources[0].trajectory is not None:
            source_pos = scene.sources[0].trajectory.location(tau_0)
        else:
            source_pos = scene.sources[0].location[:, np.newaxis]
        mic_pos = scene.microphones[0].location[:, np.newaxis]
        distance_to_mic = np.linalg.norm(source_pos - mic_pos)
        c = scene.environment.c
        return tt - distance_to_mic / c - tau_0

    num_samples = scene.sources[0].signal.num_samples
    t = np.linspace(0, 1, num_samples)
    freq = scene.sources[0].signal.freq

    sending_time = np.array([fsolve(lambda tau_0, tt=tt: arrival_time_equation(tau_0, tt), tt)[0] for tt in t])
    sending_time = np.where(sending_time < 0, 0, sending_time)

    if scene.sources[0].trajectory is not None:
        source_pos = scene.sources[0].trajectory.location(sending_time)
    else:
        source_pos = scene.sources[0].location[:, np.newaxis]
    mic_pos = scene.microphones[0].location[:, np.newaxis]
    distance = np.linalg.norm(source_pos - mic_pos, axis=0)

    solution = np.sin(2 * np.pi * freq * sending_time) / distance

    # synthesis result
    result = np.concatenate(list(scene.result(num=128))).flatten()

    msg = 'Scene synthesis result does not match analytical solution'
    np.testing.assert_allclose(result, solution, atol=1e-6, err_msg=msg)
