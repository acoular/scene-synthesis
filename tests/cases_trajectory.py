"""Test cases for source trajectories."""

import numpy as np
import scene_synthesis as ss


class Trajectories:
    """Test cases for all :class:`scene_synthesis.trajectory.FixedTrajectory` objects.

    New trajectories should be added here.
    """

    def case_none(self):
        """No trajectory test case."""
        return [None]

    def case_static(self):
        """Static trajectory test case."""
        points = {0.0: (1.0, 0.0, 0.0), 1.0: (1.0, 0.0, 0.0)}
        return [ss.FixedTrajectory(points=points)]

    def case_linear_pass(self):
        """Linear pass trajectory (fly by) test case."""
        points = {0.0: (-1.0, -1.0, 1.0), 1.0: (1.0, 1.0, 1.0)}
        return [ss.FixedTrajectory(points=points)]

    def case_linear_approach(self):
        """Linear approach trajectory (fly at) test case."""
        points = {0.0: (5.0, 0.0, 0.0), 1.0: (0.5, 0.0, 0.0)}
        return [ss.FixedTrajectory(points=points)]

    def case_circular(self):
        """Circular trajectory (fly around) test case."""
        n = 3600
        points = {i / n: (1.0 * np.cos(2 * np.pi * i / n), 1.0 * np.sin(2 * np.pi * i / n), 0.0) for i in range(n + 1)}
        return [ss.FixedTrajectory(points=points)]

    def case_static_array(self):
        """Static array trajectory test case."""
        points_array = [{0.0: (x, y, 1.0), 1.0: (x, y, 1.0)} for x, y in [(1.0, 1.0), (1.0, 0.0), (0.0, 0.0)]]
        return [ss.FixedTrajectory(points=points) for points in points_array]
