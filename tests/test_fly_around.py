import acoular as ac
import numpy as np
import scene_synthesis as synth


def test_fly_around_maneuver():
    """Test that scene synthesis matches analytical solution for fly-around maneuver."""
    # Test parameters
    c = 343.0
    T = 10
    signal_frequency = 0.5
    distance = 100.0
    num_points = 3600

    # Analytical solution for fly-around maneuver
    delay = distance / c
    t = np.linspace(0, T, num_points)
    sending_time = np.where(t - delay >= 0, t - delay, 0)
    ana_result = np.sin(2 * np.pi * signal_frequency * sending_time) / distance

    # Scene synthesis result
    theta = np.linspace(0, 2 * np.pi, num_points)
    x = distance * np.cos(theta)
    y = distance * np.sin(theta)
    z = np.zeros(num_points)
    xyz = np.vstack((x, y, z)).T

    traj = ac.Trajectory(points={time: tuple(loc) for time, loc in zip(t, xyz, strict=True)})
    gen = ac.SineGenerator(freq=signal_frequency, num_samples=num_points, sample_freq=num_points / T)

    source = synth.Source()
    source.signal = gen
    source.trajectory = traj

    mic = synth.Microphone()

    scene = synth.Scene()
    scene.environment = ac.Environment(c=c)
    scene.microphones = [mic]
    scene.sources = [source]

    synth_res = np.concatenate(list(scene.result(num=100))).flatten()

    # Assertion
    msg = 'Scene synthesis result does not match analytical solution'
    np.testing.assert_allclose(synth_res, ana_result, atol=1e-6, err_msg=msg)
