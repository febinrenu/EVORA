import subprocess
import sys
import textwrap

import numpy as np
import pytest

pytest.importorskip("scipy")

from evora.perception import lap_shim  # noqa: E402


def test_assigns_the_cheapest_matching():
    cost = np.array([[1.0, 9.0, 9.0], [9.0, 2.0, 9.0], [9.0, 9.0, 3.0]])
    total, x, y = lap_shim.lapjv(cost, extend_cost=True, cost_limit=100.0)
    assert x.tolist() == [0, 1, 2] and y.tolist() == [0, 1, 2] and total == pytest.approx(6.0)


def test_pairs_costing_more_than_the_limit_stay_unassigned():
    cost = np.array([[0.2, 0.9], [0.9, 0.95]])
    _, x, y = lap_shim.lapjv(cost, extend_cost=True, cost_limit=0.8)
    assert x.tolist() == [0, -1] and y.tolist() == [0, -1]


def test_rectangular_and_empty_inputs():
    _, x, y = lap_shim.lapjv(np.array([[0.1, 0.5, 0.9]]), extend_cost=True, cost_limit=0.8)
    assert x.tolist() == [0] and y.tolist() == [0, -1, -1]
    assert lap_shim.lapjv(np.zeros((0, 3)))[1].tolist() == []
    _, x, y = lap_shim.lapjv(np.zeros((2, 0)), extend_cost=True, cost_limit=0.5)
    assert x.tolist() == [-1, -1] and y.tolist() == []


def test_matches_the_real_lap_when_it_is_installed():
    real = pytest.importorskip("lap")
    rng = np.random.default_rng(3)
    for n, m in ((5, 5), (4, 7), (7, 4), (1, 3)):
        cost = rng.random((n, m))
        _, x_real, y_real = real.lapjv(cost, extend_cost=True, cost_limit=0.6)
        _, x_shim, y_shim = lap_shim.lapjv(cost, extend_cost=True, cost_limit=0.6)
        assert np.array_equal(x_real, x_shim) and np.array_equal(y_real, y_shim)


def test_trackers_work_in_an_environment_without_lap():
    """Run a tracker in a fresh interpreter where importing `lap` fails, as in a uv environment without pip."""
    pytest.importorskip("ultralytics")
    code = textwrap.dedent("""
        import sys
        sys.modules["lap"] = None                      # importing it raises ImportError
        import numpy as np
        from evora.perception.track import build_tracker
        tracker, _ = build_tracker("bytetrack.yaml")
        assert sys.modules["lap"].__version__.endswith("evora-shim")
        from types import SimpleNamespace
        import torch
        from ultralytics.engine.results import Boxes
        for t in range(4):
            boxes = Boxes(torch.tensor([[10.0 + 12 * t, 40, 50 + 12 * t, 160, 0.9, 0]]), (240, 320))
            out = tracker.update(boxes.cpu().numpy(), np.zeros((240, 320, 3), np.uint8))
        assert len(out) == 1 and int(out[0][4]) == 1, out
        print("ok")
    """)
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert res.returncode == 0 and "ok" in res.stdout, res.stderr[-600:]
