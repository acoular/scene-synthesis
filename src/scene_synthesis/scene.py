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

        # Carry-over receiving times and squished signals for each source-mic
        # pair. We keep a small tail of the previous block to ensure
        # numerically consistent interpolation across block boundaries.
        carry_receiving_times = [[np.array([], dtype=float) for _ in self.microphones] for _ in self.sources]
        carry_squished_signals = [[np.array([], dtype=float) for _ in self.microphones] for _ in self.sources]

        iteration = 0
        while iteration * num < num_samples:
            interpolation_space = receiving_time_space[iteration * num : (iteration + 1) * num]
            processed_signals = np.zeros((interpolation_space.size, len(self.microphones)))

            for source_id, source in enumerate(self.sources):
                for mic_id, mic in enumerate(self.microphones):
                    step = 0
                    # New samples generated in this block for this source-mic
                    new_receiving_times = np.array([], dtype=float)
                    new_distances = np.array([], dtype=float)
                    new_radial_machs = np.array([], dtype=float)
                    last_size = sent_signal_size_matrix[source_id, mic_id]

                    while not new_receiving_times.any() or new_receiving_times.max() < interpolation_space.max():
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

                        new_receiving_times = np.append(new_receiving_times, receiving_time)
                        new_distances = np.append(new_distances, distance)

                        if source.conv_amp:
                            radial_mach = np.dot(source_vel, relative_loc / distance) / c
                            new_radial_machs = np.append(new_radial_machs, radial_mach)
                        else:
                            new_radial_machs = np.append(new_radial_machs, 0.0)

                        step += 1

                    last_sending_step_matrix[source_id, mic_id] += step

                    # Fetch new signal samples for this iteration
                    signal = source.signal.signal()[last_size : last_size + new_receiving_times.size]
                    sent_signal_size_matrix[source_id, mic_id] += new_receiving_times.size

                    # Apply spherical spreading loss and Doppler effect correction
                    # Someting about the normalization factor of 4 pi is wrong.
                    # Probably has something to do with the radial Mach number.
                    new_squished_signal = signal / new_distances / np.square(1 - new_radial_machs)  # / 4 / np.pi

                    # Combine carry-over tail from previous block with newly generated samples.
                    prev_times = carry_receiving_times[source_id][mic_id]
                    prev_squished = carry_squished_signals[source_id][mic_id]
                    if prev_times.size:
                        receiving_times = np.concatenate([prev_times, new_receiving_times])
                        squished_signal = np.concatenate([prev_squished, new_squished_signal])
                    else:
                        receiving_times = new_receiving_times
                        squished_signal = new_squished_signal

                    # Interpolate signal to microphone sample times
                    interp_signal = np.interp(interpolation_space, receiving_times, squished_signal, left=0, right=0)

                    # Store a small tail (up to two samples) for the next block
                    if receiving_times.size >= 2:
                        carry_receiving_times[source_id][mic_id] = receiving_times[-2:]
                        carry_squished_signals[source_id][mic_id] = squished_signal[-2:]
                    elif receiving_times.size == 1:
                        carry_receiving_times[source_id][mic_id] = receiving_times[-1:]
                        carry_squished_signals[source_id][mic_id] = squished_signal[-1:]
                    else:
                        carry_receiving_times[source_id][mic_id] = np.array([], dtype=float)
                        carry_squished_signals[source_id][mic_id] = np.array([], dtype=float)

                    # Accumulate contributions from all sources
                    processed_signals[:, mic_id] += interp_signal

            iteration += 1
            yield processed_signals
