def pytest_addoption(parser):
    parser.addoption(
        "-D", "--debug-test", 
        action="store_true", 
        default=False,
        help="Enable breakpoint in selected tests")

import pytest

@pytest.fixture
def debug_test(request):
    return request.config.getoption("--debug-test")
