import scene_synthesis as synth
import acoular as ac
import numpy as np
import matplotlib.pyplot as plt

T = 0.5
C = 343.0
sf = 1000
ns = 1000

num = 150


# Acoular setup
mics = ac.MicGeom()
grid = ac.RectGrid(x_min=-1, x_max=1, y_min=-1, y_max=1, z=1)

traj = ac.Trajectory(points={0: (-50, 0, 1), T: (50, 0, 1)})
gen1 = ac.SineGenerator(freq=10, num_samples=ns, sample_freq=sf)
gen2 = ac.SineGenerator(freq=1, num_samples=ns, sample_freq=sf, amplitude=0.5, phase=-np.pi/2)

mps1 = ac.MovingPointSource(signal=gen1, trajectory=traj, mics=mics)
mps2 = ac.MovingPointSource(signal=gen2, trajectory=traj, mics=mics)

mix = ac.SourceMixer(sources=[mps1, mps2])

ac_res = list(mix.result(num=num))
ac_res = np.concatenate(ac_res)


# Scene Synthesis setup
source1 = synth.Source()
source1.signal = gen1
source1.trajectory = traj

source2 = synth.Source()
source2.signal = gen2
source2.trajectory = traj

mic = synth.Microphone()

scene = synth.Scene()
scene.environment = ac.Environment()
scene.microphones = [mic]
scene.sources = [source1, source2]

synth_res = np.concatenate(list(scene.result(num=num)))


# Plotting
tt = np.linspace(0, T, ns)
plt.plot(tt, ac_res, label='Acoular Result')
plt.plot(tt, synth_res, label='Scene Synthesis Result')
plt.legend()
plt.savefig('fly-by-result.png')