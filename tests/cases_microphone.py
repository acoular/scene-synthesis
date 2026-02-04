import numpy as np
import scene_synthesis as ss


class Microphones:
    def case_single(self):
        return [ss.Microphone(location=np.array((0, 0, 0)))]
