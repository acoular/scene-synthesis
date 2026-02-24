"""Scene representation with acoustic sources and microphones."""

import numpy as np
from acoular import Environment
from traits.api import CList, HasStrictTraits, Instance

from scene_synthesis.microphones import Microphone
from scene_synthesis.sources import Source


class Scene(HasStrictTraits):
    """Class representing a scene with multiple acoustic sources."""

    #: Envrionment of the scene.
    environment = Instance(Environment)

    #: List of microphones in the scene.
    microphones = CList(Instance(Microphone))

    #: List of sources in the scene.
    sources = CList(Instance(Source))

    def propagation_model(self):
        """Compute the propagation model for the scene."""
        raise NotImplementedError

    def result(self, num=128):
        """
        Generate synthesis result blockwise.

        This method performs time-domain synthesis of audio signals received at microphone
        locations. The synthesis is performed iteratively in blocks.

        Parameters
        ----------
        num:  :class:`int`, optional
            Number of samples to process per iteration. Defaults to 128.

        Yields
        ------
        :class:`numpy.ndarray`
            A 2D array of shape `(num, microphones.size)` containing the processed signal
            contributions at each microphone for the current iteration. The signal is the
            accumulated contribution from all sources after applying acoustic propagation effects.

        Notes
        -----
        - The method assumes all sources have synchronized sample frequencies.
        - Spherical spreading loss is applied as 1/distance attenuation.
        - The receiving time space is derived from the first source's signal sample frequency and
          duration, assuming all sources are synchronized.
        """
        c = self.environment.c
        sample_freq = self.sources[0].signal.sample_freq
        num_samples = self.sources[0].signal.num_samples

        t_final = num_samples / sample_freq
        receiving_time_space = np.linspace(0, t_final, num_samples)

        # min_receiving_times_matrix = np.zeros((len(self.sources), len(self.microphones)))
        # max_receiving_times_matrix = np.zeros((len(self.sources), len(self.microphones)))
        # total_sending_times = np.arange(num_samples) / sample_freq
        # for source_id, source in enumerate(self.sources):
        #     for mic_id, mic in enumerate(self.microphones):
        #         source_locs = np.array(source.trajectory.location(total_sending_times)).T
        #         relative_locs = source_locs - np.array(mic.location)
        #         distances = np.linalg.norm(relative_locs, axis=1)
        #         time_delays = distances / c
        #         min_receiving_times_matrix[source_id, mic_id] = (total_sending_times + time_delays).min() #noqa: W505
        #         max_receiving_times_matrix[source_id, mic_id] = (total_sending_times + time_delays).max() #noqa: W505
        # first_receiving_time = min_receiving_times_matrix.min()
        # final_receiving_time = max_receiving_times_matrix.max()
        # receiving_time_space = np.linspace(first_receiving_time, final_receiving_time, num_samples) #noqa: W505

        last_sending_step_matrix = np.zeros((len(self.sources), len(self.microphones)), dtype=int)
        sent_signal_size_matrix = np.zeros((len(self.sources), len(self.microphones)), dtype=int)

        # Store last receiving time and squeezed signal for each source-mic pair
        last_receiving_times = np.zeros((len(self.sources), len(self.microphones)), dtype=float)
        last_squished_signals = np.zeros((len(self.sources), len(self.microphones)), dtype=float)

        iteration = 0
        while iteration * num < num_samples:
            interpolation_space = receiving_time_space[iteration * num : (iteration + 1) * num]
            processed_signals = np.zeros((interpolation_space.size, len(self.microphones)))

            for source_id, source in enumerate(self.sources):
                for mic_id, mic in enumerate(self.microphones):
                    step = 0
                    receiving_times = np.array([])
                    distances = np.array([])
                    radial_machs = np.array([])
                    last_size = sent_signal_size_matrix[source_id, mic_id]
                    while not receiving_times.any() or receiving_times.max() < interpolation_space.max():
                        # Check if we have signal samples available
                        if last_size + step >= num_samples:
                            break

                        sending_time = (last_sending_step_matrix[source_id, mic_id] + step) / sample_freq
                        if source.trajectory is not None:
                            source_loc = np.array(source.trajectory.location(sending_time)).T
                            source_vel = np.array(source.trajectory.location(sending_time, der=1)).T
                        else:
                            source_loc = np.array(source.location)
                            source_vel = np.array([0, 0, 0])
                        relative_loc = source_loc - np.array(mic.location)
                        distance = np.linalg.norm(relative_loc)
                        time_delays = distance / c
                        receiving_time = sending_time + time_delays

                        receiving_times = np.append(receiving_times, receiving_time)
                        distances = np.append(distances, distance)

                        if source.conv_amp:
                            radial_mach = np.dot(source_vel, relative_loc / distance) / c
                            radial_machs = np.append(radial_machs, radial_mach)
                        else:
                            radial_machs = np.append(radial_machs, 0.0)

                        step += 1

                    last_sending_step_matrix[source_id, mic_id] += step

                    # Fetch new signal samples for this iteration
                    signal = source.signal.signal()[last_size : last_size + receiving_times.size]
                    sent_signal_size_matrix[source_id, mic_id] += receiving_times.size

                    # Apply spherical spreading loss and Doppler effect correction
                    # Someting about the normalization factor of 4 pi is wrong.
                    # Probably has something to do with the radial Mach number.
                    squished_signal = signal / distances / np.square(1 - radial_machs)  # / 4 / np.pi

                    # Prepend last values from previous iteration if available
                    if last_receiving_times[source_id, mic_id]:
                        receiving_times = np.concatenate([[last_receiving_times[source_id, mic_id]], receiving_times])
                        squished_signal = np.concatenate([[last_squished_signals[source_id, mic_id]], squished_signal])

                    # Interpolate signal to microphone sample times
                    interp_signal = np.interp(interpolation_space, receiving_times, squished_signal, left=0, right=0)

                    # Store last values for next iteration
                    last_sample = np.searchsorted(receiving_times, interpolation_space[-1])
                    last_receiving_times[source_id, mic_id] = receiving_times[last_sample - 1]
                    last_squished_signals[source_id, mic_id] = squished_signal[last_sample - 1]

                    # Accumulate contributions from all sources
                    processed_signals[:, mic_id] += interp_signal

            iteration += 1
            yield processed_signals
