"""Test cases for source trajectories."""

import numpy as np
import scene_synthesis as ss


class Trajectories:
    """Test cases for :class:`scene_synthesis.trajectory.SplineTrajectory` objects.

    Also goes over ``None`` (no trajectory).

    New trajectories should be added here.
    """

    def case_none(self):
        """No trajectory test case."""
        return [None]

    def case_static(self):
        """Static trajectory test case."""
        times = [0.0, 1.0]
        locations = [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
        return [ss.SplineTrajectory(times=times, locations=locations)]

    def case_linear_pass(self):
        """Linear pass trajectory (fly by) test case."""
        times = [0.0, 1.0]
        locations = [[-1.0, -1.0, 1.0], [1.0, 1.0, 1.0]]
        return [ss.SplineTrajectory(times=times, locations=locations)]

    def case_linear_approach(self):
        """Linear approach trajectory (fly at) test case."""
        times = [0.0, 1.0]
        locations = [[5.0, 0.0, 0.0], [0.5, 0.0, 0.0]]
        return [ss.SplineTrajectory(times=times, locations=locations)]

    def case_circular(self):
        """Circular trajectory (fly around) test case."""
        n = 3600
        times = np.linspace(0.0, 1.0, n + 1)
        locations = np.column_stack(
            [
                np.cos(2 * np.pi * times),
                np.sin(2 * np.pi * times),
                np.zeros_like(times),
            ]
        )
        return [ss.SplineTrajectory(times=times, locations=locations)]

    def case_static_array(self):
        """Static array trajectory test case."""
        times = [0.0, 1.0]
        static_points = [(1.0, 1.0, 1.0), (1.0, 0.0, 1.0), (0.0, 0.0, 1.0)]
        return [ss.SplineTrajectory(times=times, locations=[point, point]) for point in static_points]
