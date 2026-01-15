from acoular import Environment, Trajectory
from scene_synthesis.sources import Source
from scene_synthesis.microphones import Microphone
from traits.api import HasStrictTraits, Instance, CList
import numpy as np


class Scene(HasStrictTraits):
    """Class representing a scene with multiple acoustic sources."""

    #: Envrionment of the scene.
    environment = Instance(Environment)

    #: List of microphones in the scene.
    microphones = CList(Instance(Microphone))

    #: List of sources in the scene.
    sources = CList(Instance(Source))

    #: List of source locations in the scene.
    #: To be deleted after impelementation of trajectory class.
    trajectories = CList(Instance(Trajectory))

    def propagation_model(self):
        raise NotImplementedError

    def result(self, num=128):
        mixed_signals = np.zeros((num, len(self.microphones)))
        start_times = np.zeros((len(self.sources), len(self.microphones)))

        for source_id, (source, source_traj) in enumerate(zip(self.sources, self.trajectories)):
            c = self.environment.c
            sample_freq = source.signal.sample_freq
            signal = source.signal.signal()[:num]
            # start_times[source_id] += 1 / sample_freq
            
            for mic_id, mic in enumerate(self.microphones):
                times = start_times[source_id, mic_id] + np.arange(num) / sample_freq
                source_locs = np.array(source_traj.location(times)).T
                source_vels = np.array(source_traj.location(times, der=1)).T
                relative_locs = source_locs - np.array(mic.location)
                distances = np.linalg.norm(relative_locs, axis=1)
                time_delays = distances / c
                receiving_times = times + time_delays
                start_times[source_id, mic_id] = times[-1] - time_delays[-1]  # this part isn't right yet
                print(times)
                print(time_delays)
                print(start_times[source_id, mic_id])

                radial_Mach = 0 #-(source_vels.T * relative_locs.T).sum(0) / c / distances
                squished_signal = signal / distances / (1 - radial_Mach)**2
                interp_signal = np.interp(times, receiving_times, squished_signal[:num], left=0, right=0)
                mixed_signals[:, mic_id] += interp_signal

        return mixed_signals