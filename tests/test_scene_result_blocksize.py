"""Regression tests for blockwise scene synthesis."""

import numpy as np


def test_result_num1_matches_reference_blocksize(scene):
    """`num=1` should reproduce the same signal as a larger block size."""
    reference = np.concatenate(list(scene.result(num=128)))
    samplewise = np.concatenate(list(scene.result(num=1)))

    msg = '`Scene.result(num=1)` does not match reference block synthesis.'
    np.testing.assert_allclose(samplewise, reference, atol=1e-8, err_msg=msg)
