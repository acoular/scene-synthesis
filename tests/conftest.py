import scene_synthesis as ss
from pytest_cases import fixture, parametrize_with_cases

from tests.cases_environment import Environments
from tests.cases_microphone import Microphones
from tests.cases_source import Sources


@fixture(scope='function')
@parametrize_with_cases('environment', cases=Environments)
@parametrize_with_cases('microphones', cases=Microphones)
@parametrize_with_cases('sources', cases=Sources)
def scene(environment, microphones, sources):
    """Fixture with all considered scenes for testing."""
    return ss.Scene(environment=environment, microphones=microphones, sources=sources)
