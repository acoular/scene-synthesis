"""Regression tests for blockwise scene synthesis."""

import acoular as ac
import numpy as np
import scene_synthesis as ss


def _single_case_scene():
    """Build one representative scene for num=1 regression testing."""
    signal = ac.SineGenerator(freq=0.1, num_samples=10000, sample_freq=10000)
    source = ss.Source(signal=signal, trajectory=None, location=[1, 0, 0])
    microphone = ss.Microphone(location=np.array((0, 0, 0)))
    environment = ac.Environment(c=343.0)
    return ss.Scene(environment=environment, microphones=[microphone], sources=[source])


def test_result_num1_matches_reference_blocksize():
    """`num=1` should reproduce the same signal as a larger block size."""
    scene = _single_case_scene()
    reference = np.concatenate(list(scene.result(num=128)))
    samplewise = np.concatenate(list(scene.result(num=1)))

    msg = '`Scene.result(num=1)` does not match reference block synthesis.'
    np.testing.assert_allclose(samplewise, reference, atol=1e-8, err_msg=msg)
