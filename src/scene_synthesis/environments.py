"""Environment models for acoustic propagation."""

import numpy as np
from traits.api import CArray, Float, HasStrictTraits, Union


def dist_mat(gpos, mpos):
    """Compute the distance matrix between source and microphone positions.

    Parameters
    ----------
    gpos : :class:`numpy.ndarray`
        Positions of ``N`` points with shape ``(3, N)``.
    mpos : :class:`numpy.ndarray`
        Positions of ``M`` points with shape ``(3, M)``.

    Returns
    -------
    :class:`numpy.ndarray`
        Distance matrix with shape ``(N, M)``.
    """
    _, mics = mpos.shape
    _, points = gpos.shape
    distances = np.empty((points, mics), dtype=np.result_type(gpos, mpos, np.float64))

    for point_idx in range(points):
        point = gpos[:, point_idx]
        deltas = point[:, np.newaxis] - mpos
        distances[point_idx, :] = np.sqrt(np.sum(deltas * deltas, axis=0))

    return distances


class Environment(HasStrictTraits):
    """A simple acoustic environment without flow.

    This class mirrors the free-field behaviour needed from
    :class:`acoular.Environment` while keeping the implementation local to
    scene-synthesis.
    """

    #: Speed of sound in the environment.
    c = Float(343.0)

    #: Region of interest for calculations. Reserved for future use.
    roi = Union(None, CArray)

    def _r(self, gpos, mpos=0.0):
        """Compute apparent distances between source and microphone positions.

        Parameters
        ----------
        gpos : :class:`numpy.ndarray`
            Coordinates of the first set of points with shape ``(3, N)``.
        mpos : :class:`float` or :class:`numpy.ndarray`, optional
            Coordinates of the second set of points. A scalar is interpreted
            as the origin ``(0, 0, 0)``. Array input must have shape ``(3, M)``.

        Returns
        -------
        :class:`numpy.ndarray`
            Distances with shape ``(N,)`` for one microphone position or
            ``(N, M)`` for multiple microphone positions.
        """
        if np.isscalar(mpos):
            mpos = np.array((0.0, 0.0, 0.0), dtype=np.float64)[:, np.newaxis]

        distances = dist_mat(np.ascontiguousarray(gpos), np.ascontiguousarray(mpos))
        if distances.shape[1] == 1:
            return distances[:, 0]
        return distances

    def apparent_r(self, spos, mpos=0.0):
        """Return apparent propagation distances between source and mic positions."""
        return self._r(spos, mpos)

    def spread(self, spos, mpos=0.0):
        """Return geometric spreading factors for source-mic pairs.

        The current implementation uses simple spherical spreading, i.e.
        ``1 / apparent_r``.
        """
        distances = np.asarray(self.apparent_r(spos, mpos), dtype=float)
        return np.reciprocal(distances)
