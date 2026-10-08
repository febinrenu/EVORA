"""evora backend package."""
__version__ = "0.1.0"

import sys
from pathlib import Path

# `contracts/` lives at the repository root, outside this package; make it importable from any entry point.
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.append(_REPO_ROOT)
