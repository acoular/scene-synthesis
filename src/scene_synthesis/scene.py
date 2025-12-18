from acoular import Environment
from scene_synthesis.sources import Source
from scene_synthesis.microphones import Microphone
from traits.api import HasStrictTraits, Instance, CList


class Scene(HasStrictTraits):
    """Class representing a scene with multiple acoustic sources."""

    #: Envrionment of the scene.
    environment = Instance(Environment)

    #: List of microphones in the scene.
    microphones = CList(Instance(Microphone))

    #: List of sources in the scene.
    sources = CList(Instance(Source))

    def propagation_model(self):
        raise NotImplementedError

    def result(self, num=128):
        raise NotImplementedError
