import acoular as ac
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt



def reconstruct_signal(sig, t):
    fs = len(t) / (t[-1] - t[0])
    
    freq_sig = np.fft.rfft(sig)
    freqs = np.fft.rfftfreq(len(sig), 1/fs)
    peak_freqs, _ = sp.signal.find_peaks(np.abs(freq_sig), height=np.abs(freq_sig).max()*0.1)

    sincos = lambda b,c,f,p: [b * np.sin(2 * np.pi * f * t + p),
                              c * np.cos(2 * np.pi * f * t + p)]
    M = np.column_stack(np.concatenate([sincos(1,1,f,0) for f in freqs[peak_freqs]]))
    coeffs, _, _, _ = np.linalg.lstsq(M, sig, rcond=None)

    amplitudes = np.hypot(coeffs[0::2], coeffs[1::2])
    phases = np.arctan2(coeffs[1::2], coeffs[0::2])

    print("Frequencies:", freqs[peak_freqs])
    print("Amplitudes:", amplitudes)
    print("Phases:", phases)

    reconst_sig = lambda t: np.sum([amplitudes[i] * np.sin(2 * np.pi * freqs[peak_freqs][i] * t + phases[i]) for i in range(len(peak_freqs))], axis=0)

    return reconst_sig



T = 1
C = 343.0
fs = 1000

points = np.zeros((3, 1))
mics = ac.MicGeom(pos_total=points)
grid = ac.RectGrid(x_min=-1, x_max=1, y_min=-1, y_max=1, z=1)
traj = ac.Trajectory(points={0: (-100, 0, 1), T: (100, 0, 1)})
gen1 = ac.SineGenerator(freq=10, num_samples=1000, sample_freq=fs)
gen2 = ac.SineGenerator(freq=1, num_samples=1000, sample_freq=fs, amplitude=0.5, phase=-np.pi/2)
mps1 = ac.MovingPointSource(signal=gen1, trajectory=traj, mics=mics)
mps2 = ac.MovingPointSource(signal=gen2, trajectory=traj, mics=mics)
mix = ac.SourceMixer(sources=[mps1, mps2])

ac_res = np.concatenate(np.concatenate(list(mix.result(num=1))))
sig = np.sum([gen.signal() for gen in [gen1, gen2]], axis=0)

tt = np.linspace(0, T, ac_res.shape[0])
ttt = np.zeros(ac_res.shape[0])
ddd = np.zeros(ac_res.shape[0])

for i in range(ac_res.shape[0]):
    x,y,z = traj.location(tt[i])
    dis = np.sqrt(x**2 + y**2 + z**2)
    ttt[i] = tt[i] + dis / C
    ddd[i] = dis

freq_sig = np.fft.rfft(sig)
freqs = np.fft.rfftfreq(len(sig), d=1/fs)
reconst_sig = reconstruct_signal(sig, tt)

# amin = np.argmin(ddd)
# print(amin, ddd[amin-1:amin+2])
# lower_t = np.linspace(ttt[0], ttt[amin], len(tt[:amin+1]))
# upper_t = np.linspace(ttt[amin+1], ttt[-1], len(tt[amin+1:]))
# dopplered_t = np.concatenate([lower_t, upper_t])

fig, axes = plt.subplots(3, 1, figsize=(8, 8))
ax1, ax2, ax3 = axes

ax1.plot(tt, ac_res, label='Acoular result')
ax1.plot(ttt, sig / ddd, label='Squished signal')
ax1.plot(ttt, reconst_sig(tt) / ddd, label='Reconstructed signal')
# ax1.plot(dopplered_t, reconst_sig(tt) / ddd, label='Reconstructed signal')

# ax2.plot(tt, sig, label='Signal')
# ax2.plot(ttt, sig, label='Squished signal')
# ax2.plot(tt, reconst_sig(tt), label='Reconstructed signal')

ax2.plot(tt, tt, label='Time')
ax2.plot(tt, ttt, label='Dopplered time')
# ax2.plot(tt, dopplered_t, label='New Dopplered time')
ax2.plot(tt, ddd / 100, label='Distance')

ax3.plot(freqs, np.abs(freq_sig), label='Signal Spectrum')

for ax in axes:
    ax.hlines(0, xmin=0, xmax=T, colors='k', linestyles='dashed')
    ax.legend()
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Amplitude')

plt.savefig('test.png')