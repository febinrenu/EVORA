import subprocess
import sys


def test_app_imports_in_a_clean_interpreter(tmp_path):
    """Regression: `make dev` failed because `contracts` was only importable under pytest's pythonpath."""
    code = "import evora.api.app as a; print(a.create_app.__name__)"
    out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, cwd=tmp_path, check=False)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "create_app"
