"""Keep test runs isolated from the real pipeline state."""
import os
import tempfile
from pathlib import Path

_tmp = Path(tempfile.mkdtemp(prefix="aivm_test_"))
os.environ.setdefault("PIPELINE_STATE_FILE", str(_tmp / "pipeline.json"))
os.environ.setdefault("PIPELINE_LOG_FILE", str(_tmp / "pipeline.log"))
