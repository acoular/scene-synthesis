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
        """
        Generate audio output by propagating source signals to microphones.
        
        Args:
            num (int): Number of samples to generate per iteration.
            
        Yields:
            np.ndarray: Mixed audio signals of shape (num, len(microphones)) for each iteration.
        """
        iteration = 0
        c = self.environment.c
        sample_freq = self.sources[0].signal.sample_freq
        num_samples = self.sources[0].signal.num_samples
        t_final = num_samples / sample_freq
        print(f"[INIT] Sound speed: {c}, Sample freq: {sample_freq}, Total samples: {num_samples}, Final time: {t_final}")
        print(f"[INIT] Num sources: {len(self.sources)}, Num microphones: {len(self.microphones)}, Samples per iteration: {num}")
        # Track which source-microphone pairs are still active
        active_sources = np.ones((len(self.sources), len(self.microphones)), dtype=bool)
        # Store signal samples that couldn't be sent in the previous iteration
        signal_remaining = np.empty((len(self.sources), len(self.microphones), num), dtype=float)
        # Store distance values corresponding to remaining signal samples
        distances_remaining = np.empty((len(self.sources), len(self.microphones), num), dtype=float)
        # Store receiving times corresponding to remaining signal samples
        receiving_times_remaining = np.empty((len(self.sources), len(self.microphones), num), dtype=float)
        # Track the start time for each source-microphone pair
        start_times = np.zeros((len(self.sources), len(self.microphones)), dtype=float)
        # Track the last sample index sent for each source-microphone pair
        last_sample_sent = np.zeros((len(self.sources), len(self.microphones)), dtype=int)

        while active_sources.any():
            print(f'\n[ITER {iteration}] Starting iteration, active sources shape: {active_sources.sum()}/{active_sources.size}')
            # Initialize processed output for this iteration
            processed_signals = np.zeros((num, len(self.microphones)))

            for source_id, (source, source_traj) in enumerate(zip(self.sources, self.trajectories)):

                for mic_id, mic in enumerate(self.microphones):
                    if not active_sources[source_id, mic_id]: continue
                    print(f"  [S{source_id}->M{mic_id}] Processing source-microphone pair")

                    last_sample = last_sample_sent[source_id, mic_id]
                    # Retrieve remaining signal and distances from previous iteration
                    old_signal = signal_remaining[source_id, mic_id][:last_sample]
                    old_distances = distances_remaining[source_id, mic_id][:last_sample]
                    old_receiving_times = receiving_times_remaining[source_id, mic_id][:last_sample]
                    print(f"    Old signal size: {old_signal.shape}, old distances size: {old_distances.shape}")

                    # Fetch new signal samples for this iteration
                    new_signal = source.signal.signal()[iteration * num : (iteration + 1) * num]
                    print(f"    New signal size: {new_signal.shape}")

                    # Calculate source positions and velocities at sample times
                    times = start_times[source_id, mic_id] + np.arange(num) / sample_freq
                    source_locs = np.array(source_traj.location(times)).T
                    source_vels = np.array(source_traj.location(times, der=1)).T
                    print(f"    Source locs shape: {source_locs.shape}, min/max distance: {np.linalg.norm(source_locs - np.array(mic.location), axis=1).min():.3f}/{np.linalg.norm(source_locs - np.array(mic.location), axis=1).max():.3f}")
                    
                    # Calculate distance from source to microphone
                    relative_locs = source_locs - np.array(mic.location)
                    new_distances = np.linalg.norm(relative_locs, axis=1)
                    new_time_delays = new_distances / c
                    # Time at which the signal arrives at the microphone
                    new_receiving_times = times + new_time_delays
                    receiving_times = np.concatenate([old_receiving_times, new_receiving_times])
                    print(f"    Time delays: min={new_time_delays.min():.6f}, max={new_time_delays.max():.6f}")
                    print(f"    Receiving times size: {receiving_times.shape}, min/max: {receiving_times.min():.3f}/{receiving_times.max():.3f}")

                    # Stack old remaining samples with new samples to fill buffer
                    signal = np.concatenate([old_signal, new_signal])
                    distances = np.concatenate([old_distances, new_distances])
                    print(f"    Stacked signal size: {signal.shape}, distance size: {distances.shape}")

                    # Find the last sample that should be sent in this iteration
                    last_sample = np.searchsorted(receiving_times, times[-1])
                    last_sample_sent[source_id, mic_id] = receiving_times.size - last_sample
                    signal_sent = signal[:last_sample+1]
                    distances_sent = distances[:last_sample+1]
                    print(f"    Last sample index: {last_sample}, samples to send: {signal_sent.shape}, remaining after send: {receiving_times.size - last_sample}")

                    # Store samples that arrive after this iteration for next iteration
                    remaining_samples = signal[last_sample:]
                    if active_sources[source_id, mic_id] and len(remaining_samples) > 0:
                        signal_remaining[source_id, mic_id, :len(remaining_samples)] = remaining_samples
                        distances_remaining[source_id, mic_id, :len(remaining_samples)] = distances[last_sample:]
                    receiving_times_sent = receiving_times[:last_sample+1]
                    receiving_times_remaining[source_id, mic_id, :len(receiving_times) - last_sample] = receiving_times[last_sample:]
                    print(f"    Remaining samples for next iter: {len(remaining_samples)}")

                    # Mark source as inactive only when no new signal and remaining signal is exhausted
                    if not new_signal.any() and not remaining_samples.any():
                        active_sources[source_id, mic_id] = False
                        print(f'    [DEACTIVATED] Source {source_id} to Mic {mic_id} is inactive now.')

                    # Apply spherical spreading loss and Doppler effect correction
                    radial_Mach = 0 #-(source_vels.T * relative_locs.T).sum(0) / c / distances
                    squished_signal = signal_sent / distances_sent / (1 - radial_Mach)**2
                    
                    # Interpolate signal to microphone sample times
                    interp_signal = np.interp(times, receiving_times_sent, squished_signal, left=0, right=0)
                    # Accumulate contributions from all sources
                    processed_signals[:, mic_id] += interp_signal
                    print(f"    Interpolated signal energy: {np.sum(interp_signal**2):.4f}")

            print(f"[ITER {iteration}] Processed signals shape: {processed_signals.shape}, total energy: {np.sum(processed_signals**2):.4f}")
            iteration += 1
            yield processed_signals