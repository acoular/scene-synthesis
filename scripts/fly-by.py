import scene_synthesis as synth
import acoular as ac
import numpy as np
import matplotlib.pyplot as plt

T = 1
C = 343.0
sf = 1000
ns = 1000

num = 100

points = np.zeros((3, 1))
mics = ac.MicGeom(pos_total=points)
grid = ac.RectGrid(x_min=-1, x_max=1, y_min=-1, y_max=1, z=1)
traj = ac.Trajectory(points={0: (-50, 0, 1), T: (50, 0, 1)})
gen1 = ac.SineGenerator(freq=10, num_samples=ns, sample_freq=sf)
gen2 = ac.SineGenerator(freq=1, num_samples=ns, sample_freq=sf, amplitude=0.5, phase=-np.pi/2)
mps1 = ac.MovingPointSource(signal=gen1, trajectory=traj, mics=mics)
mps2 = ac.MovingPointSource(signal=gen2, trajectory=traj, mics=mics)
mix = ac.SourceMixer(sources=[mps1, mps2])

ac_res = list(mix.result(num=num))
ac_res = np.concatenate(ac_res)


source1 = synth.Source()
source1.signal = gen1
source1.location = np.array([0.0, 0.0, 1.0])  # unimportant for now, wating for trajectory implementation
source1.directivity = synth.OmniDirectivity()

source2 = synth.Source()
source2.signal = gen2
source2.location = np.array([0.0, 0.0, 1.0])  # unimportant for now, wating for trajectory implementation
source2.directivity = synth.OmniDirectivity()

mic = synth.Microphone()
mic.location=np.array([0.0, 0.0, 0.0])
mic.directivity=synth.OmniDirectivity()

scene = synth.Scene()
scene.environment = ac.Environment()
scene.microphones = [mic]
scene.sources = [source1, source2]
scene.trajectories = [traj, traj]

# synth_res = np.concatenate(list(scene.result(num=num)))
synth_gen = scene.result(num=num)
synth_res = next(synth_gen)
synth_res = next(synth_gen)
synth_res = next(synth_gen)

tt = np.linspace(0, T, ns)
plt.plot(tt, ac_res, label='Acoular Result')
plt.plot(np.linspace(0, T, num), synth_res, label='Scene Synthesis Result')
plt.legend()
plt.savefig('fly-by-result.png')