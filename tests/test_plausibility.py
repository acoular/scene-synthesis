"""Plausibility tests for acoustic synthesis."""

import numpy as np
from scipy.optimize import fsolve


def test_analytical(scene):
    """
    Test that the analytical solution matches the synthesis result for acoustic scenes.

    This test computes the expected microphone signals for static and moving sources analytically,
    and compares them to the synthesized output of the scene.

    Parameters
    ----------
    scene : Scene
        An instance containing sources, microphones, and environment information.

    Raises
    ------
    AssertionError
        If the analytical and synthesized results do not match within the specified tolerance.
    """

    # analytical solution
    def arrival_time_equation(tau_0, tt, src_idx, mic_idx):
        if scene.sources[src_idx].trajectory is not None:
            source_pos = scene.sources[src_idx].trajectory.location(tau_0)
        else:
            source_pos = scene.sources[src_idx].location[:, np.newaxis]
        mic_pos = scene.microphones[mic_idx].location[:, np.newaxis]
        distance_to_mic = np.linalg.norm(source_pos - mic_pos)
        c = scene.environment.c
        return tt - distance_to_mic / c - tau_0

    num_samples = scene.sources[0].signal.num_samples
    sample_freq = scene.sources[0].signal.sample_freq
    t = np.arange(num_samples) / sample_freq
    freq = scene.sources[0].signal.freq

    solutions = np.zeros((num_samples, len(scene.microphones)))
    for src_idx in range(len(scene.sources)):
        for mic_idx in range(len(scene.microphones)):
            if scene.sources[src_idx].trajectory is None:
                # Static source: use closed-form sending times instead of a root solver.
                source_pos = scene.sources[src_idx].location[:, np.newaxis]
                mic_pos = scene.microphones[mic_idx].location[:, np.newaxis]
                distance = np.linalg.norm(source_pos - mic_pos)
                c = scene.environment.c
                sending_time = t - distance / c
            else:
                # Moving source: fall back to numerical root solving for each time sample.
                sending_time = np.array([fsolve(arrival_time_equation, tt, args=(tt, src_idx, mic_idx))[0] for tt in t])
            sending_time = np.where(sending_time < 0, 0, sending_time)

            if scene.sources[src_idx].trajectory is not None:
                source_pos = scene.sources[src_idx].trajectory.location(sending_time)
            else:
                source_pos = scene.sources[src_idx].location[:, np.newaxis]
            mic_pos = scene.microphones[mic_idx].location[:, np.newaxis]
            distance = np.linalg.norm(source_pos - mic_pos, axis=0)

            solutions[:, mic_idx] += np.sin(2 * np.pi * freq * sending_time) / distance

    solution = solutions.flatten()

    # synthesis result
    result = np.concatenate(list(scene.result(num=128))).flatten()

    msg = 'Scene synthesis result does not match analytical solution'
    np.testing.assert_allclose(result, solution, atol=1e-6, err_msg=msg)
