import numpy as np
import pytest

from evora.core import cameras as cams
from evora.core import workspace as wsmod
from evora.core.db import close_all, open_db
from evora.core.vectors import open_store
from evora.memory.clarify import Clarifier
from evora.memory.embedder import DIM
from evora.memory.kb import KnowledgeBase
from evora.memory.resolve import Resolver

# phrase -> sparse concept weights. Cosine between two phrases is the dot product of their normalised vectors.
CONCEPTS: dict[str, list[tuple[int, float]]] = {
    "main gate": [(0, 1.0)],
    "front gate": [(0, 1.0), (1, 0.25)],            # cos ~0.97: above tau_hi
    "entrance": [(0, 0.75), (2, 0.66)],             # cos ~0.75: grey band
    "loading dock": [(5, 1.0)],                     # unrelated
    "side gate": [(0, 0.7), (1, 0.3), (3, 0.64)],   # between main gate and others
    "back door": [(6, 1.0), (0, 0.2)],
    "lobby": [(7, 1.0)],
    "my car": [(10, 1.0)],
    "after hours": [(20, 1.0)],
    "night shift": [(20, 0.9), (21, 0.4)],
    "gate": [(0, 1.0)],
    "gate east": [(0, 1.0), (1, 0.2)],
    "gate west": [(0, 1.0), (1, -0.2)],
}


class FakeEmbedder:
    name = "fake-test"

    def __init__(self):
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        out = np.zeros((len(texts), DIM), dtype=np.float32)
        for i, t in enumerate(texts):
            spec = CONCEPTS.get(t, [(100 + sum(map(ord, t)) % 200, 1.0)])  # unknown phrases are orthogonal-ish
            for j, w in spec:
                out[i, j] = w
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1, norms)


class Env:
    def __init__(self, root, equivalence=None, alias_embed=True):
        self.ws = wsmod.create("mem", root)
        self.db = open_db(self.ws.db_path)
        self.embedder = FakeEmbedder()
        self.store = open_store(self.ws.vectors_dir)
        self.kb = KnowledgeBase(self.db, self.store, self.embedder)
        self.resolver = Resolver(self.kb, lambda: cams.list_cameras(self.db), equivalence, alias_embed=alias_embed)
        self.clarifier = Clarifier(self.db, self.kb)

    def camera(self, name, uri=None):
        return cams.insert_camera(
            self.db, name=name, kind="file", source_uri=uri or f"/x/{name}.mp4", t0=1000.0, t0_source="manual", duration_s=60.0
        )


@pytest.fixture()
def make_env(tmp_path):
    made = []

    def build(**kw):
        env = Env(tmp_path / "ws", **kw)
        made.append(env)
        return env

    yield build
    close_all()
