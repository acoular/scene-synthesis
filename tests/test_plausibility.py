import numpy as np
import scipy.linalg as spla


def test_analytical(scene):
    """Test that analytical solution matches synthesis result."""
    # analytical solution
    num_samples = scene.sources[0].signal.num_samples
    t = np.linspace(0, 1, num_samples)
    distance = spla.norm(scene.microphones[0].location - scene.sources[0].location)
    delay = distance / scene.environment.c
    sending_time = np.where(t - delay >= 0, t - delay, 0)
    freq = scene.sources[0].signal.freq
    solution = np.sin(2 * np.pi * freq * sending_time) / distance

    # synthesis result
    result = np.concatenate(list(scene.result(num=128))).flatten()

    msg = 'Scene synthesis result does not match analytical solution'
    np.testing.assert_allclose(result, solution, atol=1e-6, err_msg=msg)
