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
        c = self.environment.c
        sample_freq = self.sources[0].signal.sample_freq
        num_samples = self.sources[0].signal.num_samples
        t_final = num_samples / sample_freq
        receiving_time_space = np.linspace(0, t_final, num_samples)

        # min_receiving_times_matrix = np.zeros((len(self.sources), len(self.microphones)))
        # max_receiving_times_matrix = np.zeros((len(self.sources), len(self.microphones)))
        # total_sending_times = np.arange(num_samples) / sample_freq
        # for source_id, (source, source_traj) in enumerate(zip(self.sources, self.trajectories)):
        #     for mic_id, mic in enumerate(self.microphones):
        #         source_locs = np.array(source_traj.location(total_sending_times)).T
        #         relative_locs = source_locs - np.array(mic.location)
        #         distances = np.linalg.norm(relative_locs, axis=1)
        #         time_delays = distances / c
        #         min_receiving_times_matrix[source_id, mic_id] = (total_sending_times + time_delays).min()
        #         max_receiving_times_matrix[source_id, mic_id] = (total_sending_times + time_delays).max()
        # first_receiving_time = min_receiving_times_matrix.min()
        # final_receiving_time = max_receiving_times_matrix.max()
        # receiving_time_space = np.linspace(first_receiving_time, final_receiving_time, num_samples)

        last_sending_step_matrix = np.zeros((len(self.sources), len(self.microphones)), dtype=int)
        sent_signal_size_matrix = np.zeros((len(self.sources), len(self.microphones)), dtype=int)


        iteration = 0
        while iteration * num < num_samples:
            
            interpolation_space = receiving_time_space[iteration * num : (iteration + 1) * num]
            processed_signals = np.zeros((interpolation_space.size, len(self.microphones)))

            for source_id, (source, source_traj) in enumerate(zip(self.sources, self.trajectories)):
                for mic_id, mic in enumerate(self.microphones):

                    step = 0
                    receiving_times = np.array([])
                    distances = np.array([])
                    while not receiving_times.any() or receiving_times.max() < interpolation_space.max():
                        sending_time = (last_sending_step_matrix[source_id, mic_id] + step) / sample_freq
                        source_locs = np.array(source_traj.location(sending_time)).T
                        # source_vels = np.array(source_traj.location(sending_time, der=1)).T
                        relative_locs = source_locs - np.array(mic.location)
                        distance = np.linalg.norm(relative_locs)
                        time_delays = distance / c
                        receiving_time = sending_time + time_delays
                        receiving_times = np.append(receiving_times, receiving_time)
                        distances = np.append(distances, distance)
                        step += 1
                    last_sending_step_matrix[source_id, mic_id] += step

                    # Fetch new signal samples for this iteration
                    last_size = sent_signal_size_matrix[source_id, mic_id]
                    signal = source.signal.signal()[last_size : last_size + receiving_times.size]
                    sent_signal_size_matrix[source_id, mic_id] += receiving_times.size
                    
                    # Apply spherical spreading loss and Doppler effect correction
                    radial_Mach = 0 #-(source_vels.T * relative_locs.T).sum(0) / c / distances
                    squished_signal = signal / distances / (1 - radial_Mach)**2
                    
                    # Interpolate signal to microphone sample times
                    interp_signal = np.interp(interpolation_space, receiving_times, squished_signal, left=0, right=0)

                    # Accumulate contributions from all sources
                    processed_signals[:, mic_id] += interp_signal

            iteration += 1
            yield processed_signals