#!/usr/bin/env python3

import acoular as ac
import numpy as np

from scene_synthesis import Source, Scene, Microphone, OmniDirectivity, CardioidDirectivity


microphones = [
    Microphone(
        location=np.array([0, 5, 0]),
        directivity=CardioidDirectivity()
    )
    for x in (-1, 0, 1)
]

sources = [
    Source(
        signal=ac.SineGenerator(freq=f, num_samples=48000, sample_freq=48000),
        location=np.array([x, 0, 0]),
        directivity=OmniDirectivity()
    )
    for (f, x) in zip((200, 600, 1200), (-2, 0, 2), strict=True)
]

scene = Scene(microphones=microphones, sources=sources)
