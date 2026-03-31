"""Scene representation with acoustic sources and microphones."""

import numpy as np
from scipy.interpolate import CubicSpline
from traits.api import CList, HasStrictTraits, Instance

from scene_synthesis.environments import Environment
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
        """Compute the propagation model for the scene."""
        raise NotImplementedError

    def _interpolate_block_signal(self, interpolation_space, receiving_times, squished_signal):
        """Interpolate one propagated signal block onto microphone sample times.

        A cubic spline is used when enough support points are available,
        otherwise linear interpolation is used as a fallback.
        """
        if receiving_times.size >= 4:
            spline = CubicSpline(receiving_times, squished_signal, extrapolate=False)
            interp_signal = spline(interpolation_space)
            return np.where(np.isnan(interp_signal), 0.0, interp_signal)

        return np.interp(interpolation_space, receiving_times, squished_signal, left=0.0, right=0.0)

    def _source_state(self, source, sending_time):
        """Return source position and velocity at one sending time."""
        if source.trajectory is not None:
            source_loc = np.array(source.trajectory.location(sending_time)).T
            source_vel = np.array(source.trajectory.location(sending_time, der=1)).T
        else:
            source_loc = np.array(source.location)
            source_vel = np.array([0, 0, 0])

        return source_loc, source_vel

    def _propagation_sample(self, source, source_loc, source_vel, mic):
        """Return propagation quantities for one source-microphone sample."""
        c = self.environment.c
        mic_loc = np.array(mic.location)
        relative_loc = source_loc - mic_loc
        source_pos = np.asarray(source_loc, dtype=float).reshape(3, -1)
        distance = float(np.asarray(self.environment.apparent_r(source_pos, mic_loc)).reshape(-1)[0])
        spread = float(np.asarray(self.environment.spread(source_pos, mic_loc)).reshape(-1)[0])
        radial_mach = float(np.dot(source_vel, relative_loc / distance) / c) if source.conv_amp else 0.0
        return distance, spread, radial_mach

    def result(self, num=128):
        """
        Generate synthesis result block-wise.

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

        receiving_time_space = np.arange(num_samples) / sample_freq

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

            next_block_start_idx = (iteration + 1) * num
            has_next_block = next_block_start_idx < receiving_time_space.size
            next_block_start = receiving_time_space[next_block_start_idx] if has_next_block else None

            for source_id, source in enumerate(self.sources):
                for mic_id, mic in enumerate(self.microphones):
                    step = 0
                    # New samples generated in this block for this source-mic
                    # pair. Use Python lists here to avoid repeated array
                    # reallocations from ``np.append`` in the inner loop.
                    new_receiving_times_list: list[float] = []
                    new_spreads_list: list[float] = []
                    new_radial_machs_list: list[float] = []
                    last_size = sent_signal_size_matrix[source_id, mic_id]

                    while not new_receiving_times_list or new_receiving_times_list[-1] < interpolation_space.max():
                        # Check if we have signal samples available
                        if last_size + step >= num_samples:
                            break

                        sending_time = (last_sending_step_matrix[source_id, mic_id] + step) / sample_freq
                        source_loc, source_vel = self._source_state(source, sending_time)
                        distance, spread, radial_mach = self._propagation_sample(source, source_loc, source_vel, mic)
                        time_delays = distance / c
                        receiving_time = sending_time + time_delays

                        new_receiving_times_list.append(float(receiving_time))
                        new_spreads_list.append(spread)
                        new_radial_machs_list.append(radial_mach)

                        step += 1

                    new_receiving_times = np.array(new_receiving_times_list, dtype=float)
                    new_spreads = np.array(new_spreads_list, dtype=float)
                    new_radial_machs = np.array(new_radial_machs_list, dtype=float)

                    last_sending_step_matrix[source_id, mic_id] += step

                    # Fetch new signal samples for this iteration
                    signal = source.signal.signal()[last_size : last_size + new_receiving_times.size]
                    sent_signal_size_matrix[source_id, mic_id] += new_receiving_times.size

                    # Apply geometric spreading from the environment and
                    # Doppler effect correction.
                    # Something about the normalization factor of 4 pi is wrong.
                    # Probably has something to do with the radial Mach number.
                    new_squished_signal = signal * new_spreads / np.square(1 - new_radial_machs)  # / 4 / np.pi

                    # Combine carry-over tail from previous block with newly generated samples.
                    prev_times = carry_receiving_times[source_id][mic_id]
                    prev_squished = carry_squished_signals[source_id][mic_id]
                    if prev_times.size:
                        receiving_times = np.concatenate([prev_times, new_receiving_times])
                        squished_signal = np.concatenate([prev_squished, new_squished_signal])
                    else:
                        receiving_times = new_receiving_times
                        squished_signal = new_squished_signal

                    # Interpolate signal to microphone sample times.
                    interp_signal = self._interpolate_block_signal(
                        interpolation_space, receiving_times, squished_signal
                    )

                    # Keep only the carry-over points that are still relevant for
                    # the next interpolation block. For correctness, retain the
                    # sample directly before the next block start (if available)
                    # plus all later samples.
                    if not has_next_block:
                        carry_receiving_times[source_id][mic_id] = np.array([], dtype=float)
                        carry_squished_signals[source_id][mic_id] = np.array([], dtype=float)
                    elif receiving_times.size:
                        first_future_idx = np.searchsorted(receiving_times, next_block_start, side='left')
                        keep_from = max(first_future_idx - 1, 0)
                        carry_receiving_times[source_id][mic_id] = receiving_times[keep_from:]
                        carry_squished_signals[source_id][mic_id] = squished_signal[keep_from:]
                    else:
                        carry_receiving_times[source_id][mic_id] = np.array([], dtype=float)
                        carry_squished_signals[source_id][mic_id] = np.array([], dtype=float)

                    # Accumulate contributions from all sources
                    processed_signals[:, mic_id] += interp_signal

            iteration += 1
            yield processed_signals
