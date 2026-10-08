import shutil
import subprocess
from pathlib import Path

import pytest

from evora.core import db as dbmod


@pytest.fixture(autouse=True)
def _close_dbs():
    yield
    dbmod.close_all()


@pytest.fixture(scope="session")
def sample_mp4(tmp_path_factory) -> Path:
    """A 2 s, 320x240, 10 fps test clip made with ffmpeg's built-in source."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    out = tmp_path_factory.mktemp("clips") / "sample.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=10:duration=2",
         "-pix_fmt", "yuv420p", "-metadata", "creation_time=2026-10-08T09:00:00Z", str(out)],
        check=True,
    )
    return out
