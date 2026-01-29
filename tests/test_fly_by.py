import acoular as ac
import matplotlib.pyplot as plt
import numpy as np
import scene_synthesis as synth


def test_fly_by_maneuver():
    """Test that scene synthesis matches analytical solution for fly-by maneuver."""
    # Test parameters
    c = 343.0
    T = 10
    signal_frequency = 1
    path_length = 1000.0
    num_points = 10000

    # Analytical solution for fly-by maneuver
    # The microphone is at the origin (0,0,0)
    # The source moves along the x-axis from (-path_length/2, 0, 1) to (path_length/2, 0, 1)
    t = np.linspace(0, T, num_points)
    sending_times = (c * t - path_length / 2) / (c + path_length / T)
    sending_times = np.where(sending_times < 0, 0, sending_times)
    distances = np.sqrt((sending_times * (path_length / T) - path_length / 2) ** 2 + 1)
    ana_result = np.sin(2 * np.pi * signal_frequency * sending_times) / distances

    # Scene synthesis result
    traj = ac.Trajectory(points={0: (-path_length / 2, 0, 1), T: (path_length / 2, 0, 1)})
    gen = ac.SineGenerator(freq=signal_frequency, num_samples=num_points, sample_freq=num_points / T)

    source = synth.Source()
    source.signal = gen
    source.trajectory = traj

    mic = synth.Microphone()

    scene = synth.Scene()
    scene.environment = ac.Environment(c=c)
    scene.microphones = [mic]
    scene.sources = [source]

    synth_result = np.concatenate(list(scene.result(num=100))).flatten()

    plt.plot(t, ana_result, label='Analytical Solution')
    plt.plot(t, synth_result, label='Scene Synthesis')
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Fly-By Maneuver')
    plt.savefig('test_fly_by_maneuver.png')

    # Assertion
    msg = 'Scene synthesis result does not match analytical solution'
    np.testing.assert_allclose(synth_result, ana_result, err_msg=msg)
