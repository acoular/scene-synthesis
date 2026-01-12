from pathlib import Path

import acoular as ac
import numpy as np
import matplotlib.pyplot as plt


T = 1
C = 343.0


points = np.zeros((3, 1))
mics = ac.MicGeom(pos_total=points)

grid = ac.RectGrid(x_min=-1, x_max=1, y_min=-1, y_max=1, z=1)

traj = ac.Trajectory(points={0: (-100, 0, 1), T: (100, 0, 1)})

gen = ac.SineGenerator(freq=10, num_samples=1000, sample_freq=1000)

mps = ac.MovingPointSource(signal=gen, trajectory=traj, mics=mics)

ac_res = np.concatenate(np.concatenate(list(mps.result(num=1))))


tt = np.linspace(0, T, ac_res.shape[0])
ttt = np.zeros(ac_res.shape[0])
for i in range(ac_res.shape[0]):
    x,y,z = traj.location(tt[i])
    dis = np.sqrt(x**2 + y**2 + z**2)
    ttt[i] = tt[i] + dis / C
plt.plot(tt, ac_res)
plt.plot(ttt, gen.signal())
plt.hlines(0, xmin=0, xmax=T, colors='k', linestyles='dashed')
plt.savefig('test.png')