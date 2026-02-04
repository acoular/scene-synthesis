import acoular as ac
from pytest_cases import parametrize


class Environments:
    """Test cases for all environments.

    New environments should be added here.
    """

    @parametrize('c', [343.0, 300.0])
    def case_free_field(self, c):
        return ac.Environment(c=c)

    # def case_half_space(self):
    #     pass
