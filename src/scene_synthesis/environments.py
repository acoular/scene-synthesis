"""Environment models for acoustic propagation."""

import numpy as np
from acoular import Environment as AcoularEnvironment


class Environment(AcoularEnvironment):
    """Scene-synthesis environment.

    This class currently extends :class:`acoular.Environment` with a small,
    scene-synthesis-oriented API for querying apparent propagation distances
    and geometric spreading factors.
    """

    def apparent_r(self, spos, mpos=0.0):
        """Return apparent propagation distances between source and mic positions.

        Parameters
        ----------
        spos : :class:`numpy.ndarray`
            Source positions with shape ``(3, N)``.
        mpos : :class:`float` or :class:`numpy.ndarray`, optional
            Microphone positions with shape ``(3, M)``. A scalar is treated as
            the origin, matching :class:`acoular.Environment` behaviour.

        Returns
        -------
        :class:`numpy.ndarray`
            Apparent propagation distances. The shape matches the behaviour of
            :meth:`acoular.Environment._r`.
        """
        return self._r(spos, mpos)

    def spread(self, spos, mpos=0.0):
        """Return the geometric spreading factor for source-mic pairs.

        The current implementation uses simple spherical spreading, i.e.
        ``1 / apparent_r``.
        """
        distances = np.asarray(self.apparent_r(spos, mpos), dtype=float)
        return np.reciprocal(distances)
