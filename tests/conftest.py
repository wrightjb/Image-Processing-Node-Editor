import os
import sys
from pathlib import Path

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "-D", "--debug-test",
        action="store_true",
        default=False,
        help="Enable breakpoint in selected tests",
    )
    parser.addoption(
        "--use-cv2-stub",
        action="store_true",
        default=False,
        help=(
            "Use tests/stubs/cv2.py instead of system OpenCV. "
            "Useful in environments missing libGL.so.1."
        ),
    )


@pytest.fixture
def debug_test(request):
    return request.config.getoption("--debug-test")


def _use_cv2_stub(config):
    return config.getoption("--use-cv2-stub") or os.environ.get(
        "USE_CV2_STUB", ""
    ).lower() in {"1", "true", "yes", "on"}


def pytest_configure(config):
    if not _use_cv2_stub(config):
        return

    stubs_dir = Path(__file__).parent / "stubs"
    sys.path.insert(0, str(stubs_dir))
    # Ensure import cv2 resolves to the stub even if cv2 was previously imported.
    sys.modules.pop("cv2", None)


def pytest_collection_modifyitems(config, items):
    if not _use_cv2_stub(config):
        return

    skip_integration = pytest.mark.skip(
        reason="integration tests require real OpenCV; running with cv2 stub"
    )
    for item in items:
        if "tests/integration/" in str(item.fspath):
            item.add_marker(skip_integration)
