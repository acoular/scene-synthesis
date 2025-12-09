import acoular as ac
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as spla
from scipy.interpolate import CubicSpline


LX = 50.0  # m, distance to start position on x-axis
LZ = 10.0  # m, distance between array and source plane
V_KMH = 300.0  # km/h, source velocity
VX_M = V_KMH / 3.6  # m/s, source velocity converted to meters per second
T_PASS = 2 * LX / VX_M  # s, total pass-by duration
FS = 12000  # Hz, sampling frequency
NUM_SAMPLES = int(T_PASS * FS)  # total number of samples
STIME = np.arange(NUM_SAMPLES)/FS  # source time
C = 343.0

s0, s1 = (-LX, 0, LZ), (LX, 0, LZ)
tr0 = ac.Trajectory(points={0: s0, T_PASS: s1})
s0, s1 = np.array(s0).reshape(3,1), np.array(s1).reshape(3,1)
r0 = np.array([0,1,0]).reshape(3,1)
DIST = spla.norm(np.stack(tr0.location(STIME))-r0, axis=0)  # source receiver distances (at source time)
START = DIST[0] / C  # time offset first sample
RTIME_EQ = STIME + START  # equidistant receiver time

# ACOULAR REFERENCE
mics = ac.MicGeom(pos_total=r0)
sig = ac.SineGenerator(num_samples=NUM_SAMPLES, freq=100, sample_freq=FS)
mps = ac.MovingPointSource(signal=sig, mics=mics, trajectory=tr0)
mps.start = START
ref = ac.tools.return_result(mps)

print(START)
print(NUM_SAMPLES, len(ref))

SIGNAL = sig.signal()

# SOURCE TIME SYNTHESIS
rtime = STIME + DIST/343.0
cs = CubicSpline(rtime, SIGNAL/DIST)

# PLOTTING
fig, ax = plt.subplots(2, dpi=200, sharex=True, constrained_layout=True)
ax[0].plot(STIME, SIGNAL)
ax[1].plot(rtime[:len(ref)], ref)
plt.savefig('comparison.png')

fig, ax = plt.subplots(2, dpi=200, sharex=True, constrained_layout=True)
ax[0].plot(STIME, SIGNAL)
ax[1].plot(RTIME_EQ, cs(RTIME_EQ))
plt.savefig('source_time.png')
