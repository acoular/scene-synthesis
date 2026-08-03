"""Unit tests for acoustic source models."""

import numpy as np
import acoular as ac
import scene_synthesis as ss
from traits.api import CArray


class ArraySignal(ac.SignalGenerator):
    """Signal generator backed by fixed samples."""

    samples = CArray(dtype=float)

    def _get_digest(self):
        """Return stable test digest."""
        return 'array-signal'

    def signal(self):
        """Return fixed signal samples."""
        return self.samples


class SilentLocalSource(ss.Source):
    """Source with nonzero raw signal but zero local strength."""

    def local_strength(self, environment):  # noqa: ARG002
        """Return zero source strength."""
        return np.zeros(self.signal.num_samples)


def spherical_surface(radius, n_theta=25, n_phi=50):
    """Return points and area weights for midpoint integration on a sphere."""
    theta_edges = np.linspace(0.0, np.pi, n_theta + 1)
    phi_edges = np.linspace(0.0, 2.0 * np.pi, n_phi + 1)
    theta = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    phi = 0.5 * (phi_edges[:-1] + phi_edges[1:])
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')

    points = radius * np.vstack(
        [
            (np.sin(theta_grid) * np.cos(phi_grid)).ravel(),
            (np.sin(theta_grid) * np.sin(phi_grid)).ravel(),
            np.cos(theta_grid).ravel(),
        ]
    )
    dtheta = np.diff(theta_edges)[:, np.newaxis]
    dphi = np.diff(phi_edges)[np.newaxis, :]
    weights = (radius**2 * np.sin(theta_grid) * dtheta * dphi).ravel()
    return points, weights


def propagate_periodic(source_strength, distances, sample_freq, sound_speed):
    """Apply periodic retarded-time propagation for test signals."""
    spectrum = np.fft.rfft(source_strength)
    frequencies = np.fft.rfftfreq(source_strength.size, 1.0 / sample_freq)
    phase = np.exp(-2j * np.pi * frequencies[:, np.newaxis] * distances[np.newaxis, :] / sound_speed)
    return np.fft.irfft(spectrum[:, np.newaxis] * phase, n=source_strength.size, axis=0)


def test_monopole_local_strength_is_source_time_derivative():
    """Monopole local strength should be rho0/(4*pi) times dQ/dt_s."""
    q = np.array([0.0, 1.0, 3.0, 6.0])
    sample_freq = 2.0
    environment = ss.Environment(rho0=1.5)
    signal = ArraySignal(samples=q, sample_freq=sample_freq, num_samples=q.size)
    source = ss.MonopoleSource(signal=signal)

    local_strength = source.local_strength(environment)

    expected = environment.rho0 / (4 * np.pi) * np.diff(q, prepend=0.0) * sample_freq
    np.testing.assert_allclose(local_strength, expected)


def test_monopole_local_strength_scales_with_environment_density():
    """Monopole local strength should use the active environment density."""
    q = np.array([0.0, 1.0, 2.0])
    signal = ArraySignal(samples=q, sample_freq=1.0, num_samples=q.size)
    source = ss.MonopoleSource(signal=signal)

    low_density = source.local_strength(ss.Environment(rho0=1.0))
    high_density = source.local_strength(ss.Environment(rho0=2.0))

    np.testing.assert_allclose(high_density, 2.0 * low_density)


def test_scene_uses_local_strength_instead_of_raw_signal():
    """Scene synthesis should propagate local strength, not raw signal."""
    num_samples = 256
    sample_freq = 1024.0
    time = np.arange(num_samples) / sample_freq
    raw_signal = np.sin(2 * np.pi * 64.0 * time)
    signal = ArraySignal(samples=raw_signal, sample_freq=sample_freq, num_samples=num_samples)
    source = SilentLocalSource(signal=signal, location=[1.0, 0.0, 0.0])
    microphone = ss.Microphone(location=np.array([0.0, 0.0, 0.0]))
    scene = ss.Scene(environment=ss.Environment(), microphones=[microphone], sources=[source])

    result = np.concatenate(list(scene.result(num=64)))

    np.testing.assert_allclose(result, 0.0)


def test_scene_applies_source_direction_factor():
    """Scene synthesis should apply dipole direction factors during propagation."""
    num_samples = 512
    sample_freq = 2048.0
    time = np.arange(num_samples) / sample_freq
    force = np.sin(2 * np.pi * 64.0 * time)
    signal = ArraySignal(samples=force, sample_freq=sample_freq, num_samples=num_samples)
    source = ss.DipoleSource(signal=signal, location=[0.0, 0.0, 0.0])
    environment = ss.Environment()

    perpendicular_mic = ss.Microphone(location=np.array([1.0, 0.0, 0.0]))
    perpendicular_scene = ss.Scene(environment=environment, microphones=[perpendicular_mic], sources=[source])
    perpendicular = np.concatenate(list(perpendicular_scene.result(num=64)))

    axial_mic = ss.Microphone(location=np.array([0.0, 0.0, 1.0]))
    axial_scene = ss.Scene(environment=environment, microphones=[axial_mic], sources=[source])
    axial = np.concatenate(list(axial_scene.result(num=64)))

    np.testing.assert_allclose(perpendicular, 0.0, atol=1e-12)
    assert np.max(np.abs(axial)) > 0.0


def test_monopole_finite_volume_velocity_power_integrates_on_sphere():
    """Propagated pressure on a sphere should integrate to monopole power."""
    sample_freq = 8192.0
    num_samples = 8192
    source_frequency = 64.0
    radius = 2.0
    time = np.arange(num_samples) / sample_freq
    q = np.sin(2 * np.pi * source_frequency * time)
    environment = ss.Environment(rho0=1.2041, c=343.0)
    signal = ArraySignal(samples=q, sample_freq=sample_freq, num_samples=num_samples)
    source = ss.MonopoleSource(signal=signal)

    source_strength = source.local_strength(environment)
    points, area_weights = spherical_surface(radius)
    distances = environment.apparent_r(np.array([[0.0], [0.0], [0.0]]), points).ravel()
    spread = environment.spread(np.array([[0.0], [0.0], [0.0]]), points).ravel()
    pressure = propagate_periodic(source_strength, distances, sample_freq, environment.c) * spread
    mean_square_pressure = np.mean(pressure**2, axis=0)
    numeric_radiated_power = np.sum(area_weights * mean_square_pressure / (environment.rho0 * environment.c))

    wave_number = 2 * np.pi * source_frequency / environment.c
    expected_radiated_power = np.mean(q**2) * environment.rho0 * environment.c * wave_number**2 / (4 * np.pi)

    assert np.all(np.isfinite(pressure))
    assert np.allclose(distances, radius)
    np.testing.assert_allclose(area_weights.sum(), 4 * np.pi * radius**2, rtol=7e-4)
    np.testing.assert_allclose(numeric_radiated_power, expected_radiated_power, rtol=2e-3)
