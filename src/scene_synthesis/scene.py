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
        """Generate acoustic simulation result.
        
        Parameters
        ----------
        num : int, optional
            Number of sample blocks to simulate. Default is 128.
            
        Returns
        -------
        ndarray
            Mixed acoustic signal from all sources at microphone locations.
        """
        mixed_signals = np.zeros((len(self.microphones), num))

        for source, source_traj in zip(self.sources, self.trajectories):
            signal = source.signal.signal()
            times = np.linspace(0, 1, num)  # provisionally assuming 1 second duration
            for n, mic in enumerate(self.microphones):
                source_locs = np.array(source_traj.location(times))
                distances = np.linalg.norm(source_locs.T - np.array(mic.location), axis=1)
                time_delays = distances / self.environment.c
                receiving_times = times + time_delays
                squished_signal = np.interp(times, receiving_times, signal[:num] / distances, left=0, right=0)
                mixed_signals[n] += squished_signal

        return mixed_signals