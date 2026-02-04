import acoular as ac
import numpy as np


class Trajectories:
    """Test cases for all :class:`acoular.trajectories.Trajectory`-derived classes.

    New trajectories should be added here.
    """

    def case_none(self):
        return None

    def case_static(self):
        points = {0.0: (0.0, 0.0, 0.0), 1.0: (0.0, 0.0, 0.0)}
        return ac.Trajectory(points=points)

    def case_linear(self):
        points = {0.0: (-1.0, -1.0, 1.0), 1.0: (1.0, 1.0, 1.0)}
        return ac.Trajectory(points=points)

    def case_circular(self):
        n = 3600
        points = {i / n: (1.0 * np.cos(2 * np.pi * i / n), 1.0 * np.sin(2 * np.pi * i / n), 0.0) for i in range(n + 1)}
        return ac.Trajectory(points=points)
