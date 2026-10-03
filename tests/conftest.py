"""Keep test runs isolated from the real pipeline state."""
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_tmp = Path(tempfile.mkdtemp(prefix="aivm_test_"))
os.environ.setdefault("PIPELINE_STATE_FILE", str(_tmp / "pipeline.json"))
os.environ.setdefault("PIPELINE_LOG_FILE", str(_tmp / "pipeline.log"))

# Never sleep during tests (image throttling / retry backoff are real waits).
os.environ.setdefault("FREE_AI_IMAGE_DELAY", "0")
os.environ.setdefault("FREE_AI_IMAGE_BACKOFF", "0")


@pytest.fixture(autouse=True)
def _reset_image_cooldown():
    """The 402 cooldown is process-global; don't leak it between tests."""
    from app import free_ai

    free_ai._ENDPOINT_COOLDOWN.clear()
    yield
    free_ai._ENDPOINT_COOLDOWN.clear()
