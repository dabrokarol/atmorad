from pathlib import Path

import matplotlib
import pytest

from atmorad.config.loader import load_scenarios

matplotlib.use("Agg")


@pytest.fixture
def config_list(request):
    filename = request.param
    config_path = Path(__file__).parent / filename
    return load_scenarios(config_path)
