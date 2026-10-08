import shutil
import subprocess
from pathlib import Path

import pytest

from evora.core import db as dbmod

REAL_PERCEPTION_PREFIXES = ("evora.perception", "evora.reid")
HEAVY = ("ingest", "live_ingest", "query_embedder", "link_global_ids", "similar_tracks")  # need torch, models or a GPU


@pytest.fixture(autouse=True)
def _simulated_perception(request, monkeypatch):
    """Platform tests must not load the real (slow, GPU) perception stack, even when it is installed.

    The heavy functions the adapter would find in `evora.perception` or `evora.reid` are treated as absent, so the same tests give
    the same answer with and without the stack. Fakes that tests install themselves are untouched. A test that wants
    the real functions says so with `@pytest.mark.real_perception`.
    """
    if request.node.get_closest_marker("real_perception"):
        yield
        return
    from evora.core import perception_adapter

    real_find = perception_adapter._find

    def find(name):
        fn = real_find(name)
        if name in HEAVY and fn is not None and getattr(fn, "__module__", "").startswith(REAL_PERCEPTION_PREFIXES):
            return None
        return fn

    monkeypatch.setattr(perception_adapter, "_find", find)
    yield


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


@pytest.fixture(scope="session")
def mpeg2_avi(tmp_path_factory) -> Path:
    """A clip in a codec the pipeline does not read natively (stands in for Indeo and friends)."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    out = tmp_path_factory.mktemp("odd") / "legacy.avi"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=10:duration=2",
         "-c:v", "mpeg2video", "-metadata", "creation_time=2026-10-08T09:00:00Z", str(out)],
        check=True,
    )
    return out


@pytest.fixture(scope="session")
def sample_mp4_b(tmp_path_factory) -> Path:
    """A second, different clip (so two uploads are not de-duplicated)."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    out = tmp_path_factory.mktemp("clips_b") / "sample_b.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=10:duration=3",
         "-pix_fmt", "yuv420p", "-metadata", "creation_time=2026-10-08T09:00:00Z", str(out)],
        check=True,
    )
    return out
