import numpy as np
from acoular import Environment
from traits.api import CList, HasStrictTraits, Instance

from scene_synthesis.microphones import Microphone
from scene_synthesis.sources import Source


class Scene(HasStrictTraits):
    """Class representing a scene with multiple acoustic sources."""

    #: Environment of the scene.
    environment = Instance(Environment)

    #: List of microphones in the scene.
    microphones = CList(Instance(Microphone))

    #: List of sources in the scene.
    sources = CList(Instance(Source))

    def propagation_model(self):
        raise NotImplementedError

    def _calculate_receiving_times(self):
        """
        Calculate the receiving times at the microphones.

        Returns a time space from 0 to the sources' final sending time.
        Most likely, no signal will be mapped entirely to this time space.
        """
        sample_freq = self.sources[0].signal.sample_freq
        num_samples = self.sources[0].signal.num_samples

        t_final = num_samples / sample_freq
        receiving_time_space = np.linspace(0, t_final, num_samples)

        return receiving_time_space

    # Not used currently, but might be useful in future design choices.
    def _calculate_alt_receiving_times(self):
        """
        Calculate the receiving times at each microphone for all sources.

        Returns a time space that covers all receiving times.
        """
        sample_freq = self.sources[0].signal.sample_freq
        num_samples = self.sources[0].signal.num_samples

        min_receiving_times_matrix = np.zeros((len(self.sources), len(self.microphones)))
        max_receiving_times_matrix = np.zeros((len(self.sources), len(self.microphones)))
        total_sending_times = np.arange(num_samples) / sample_freq
        for source_id, source in enumerate(self.sources):
            for mic_id, mic in enumerate(self.microphones):
                source_locs = np.array(source.trajectory.location(total_sending_times)).T
                relative_locs = source_locs - np.array(mic.location)
                distances = np.linalg.norm(relative_locs, axis=1)
                time_delays = distances / self.environment.c
                min_receiving_times_matrix[source_id, mic_id] = (total_sending_times + time_delays).min()
                max_receiving_times_matrix[source_id, mic_id] = (total_sending_times + time_delays).max()
        first_receiving_time = min_receiving_times_matrix.min()
        final_receiving_time = max_receiving_times_matrix.max()
        receiving_time_space = np.linspace(first_receiving_time, final_receiving_time, num_samples)

        return receiving_time_space

    def _setup_new_iteration(self, source, mic, last_sending_step, final_receiving_time):
        step = 0
        receiving_times = np.array([])
        distances = np.array([])
        radial_Machs = np.array([])
        while not receiving_times.any() or receiving_times.max() < final_receiving_time:
            sending_time = (last_sending_step + step) / source.signal.sample_freq
            source_loc = np.array(source.trajectory.location(sending_time)).T
            source_vel = np.array(source.trajectory.location(sending_time, der=1)).T
            relative_loc = source_loc - np.array(mic.location)
            distance = np.linalg.norm(relative_loc)
            time_delays = distance / self.environment.c
            receiving_time = sending_time + time_delays

            receiving_times = np.append(receiving_times, receiving_time)
            distances = np.append(distances, distance)

            if source.conv_amp:
                radial_Mach = np.dot(source_vel, relative_loc / distance) / self.environment.c
                radial_Machs = np.append(radial_Machs, radial_Mach)
            else:
                radial_Machs = np.append(radial_Machs, 0.0)

            step += 1

        return receiving_times, distances, radial_Machs, step

    def _interpolate_signal(self, times_wanted, times_available, signal_available):
        return np.interp(times_wanted, times_available, signal_available, left=0, right=0)

    def result(self, num=128):
        receiving_time_space = self._calculate_receiving_times()

        # Matrices to keep track of iteration state for each source-microphone pair
        last_sending_step_matrix = np.zeros((len(self.sources), len(self.microphones)), dtype=int)
        sent_signal_size_matrix = np.zeros((len(self.sources), len(self.microphones)), dtype=int)
        last_receiving_times_matrix = np.zeros((len(self.sources), len(self.microphones)), dtype=float)
        last_squished_signals_matrix = np.zeros((len(self.sources), len(self.microphones)), dtype=float)

        iteration = 0
        while iteration * num < receiving_time_space.size:
            interpolation_space = receiving_time_space[iteration * num : (iteration + 1) * num]
            processed_signals = np.zeros((interpolation_space.size, len(self.microphones)))

            for source_id, source in enumerate(self.sources):
                for mic_id, mic in enumerate(self.microphones):
                    last_sending_step = last_sending_step_matrix[source_id, mic_id]
                    final_receiving_time = interpolation_space.max()
                    receiving_times, distances, radial_Machs, last_step = self._setup_new_iteration(
                        source, mic, last_sending_step, final_receiving_time
                    )

                    # Fetch new signal samples for this iteration
                    last_size = sent_signal_size_matrix[source_id, mic_id]
                    signal = source.signal.signal()[last_size : last_size + receiving_times.size]
                    # Apply spherical spreading loss and Doppler effect correction
                    # Someting about the normalization factor of 4 pi is wrong.
                    # Probably has something to do with the radial Mach number.
                    squished_signal = signal / distances / np.square(1 - radial_Machs)  # / 4 / np.pi

                    # Update sent signal size matrix
                    sent_signal_size_matrix[source_id, mic_id] += receiving_times.size

                    # Prepend last values from previous iteration if available
                    if last_receiving_times_matrix[source_id, mic_id]:
                        receiving_times = np.concatenate(
                            [[last_receiving_times_matrix[source_id, mic_id]], receiving_times]
                        )
                        squished_signal = np.concatenate(
                            [[last_squished_signals_matrix[source_id, mic_id]], squished_signal]
                        )

                    # Interpolate signal to microphone sample times
                    interp_signal = self._interpolate_signal(interpolation_space, receiving_times, squished_signal)

                    # Update iteration state matrices
                    last_sending_step_matrix[source_id, mic_id] += last_step
                    last_sample = np.searchsorted(receiving_times, interpolation_space[-1])
                    last_receiving_times_matrix[source_id, mic_id] = receiving_times[last_sample - 1]
                    last_squished_signals_matrix[source_id, mic_id] = squished_signal[last_sample - 1]

                    # Accumulate contributions from all sources
                    processed_signals[:, mic_id] += interp_signal

            iteration += 1
            yield processed_signals
