"""Query by example: tracks that look like a given track, from the ReID vectors."""
from __future__ import annotations

import re

from evora.core.vectors import open_store
from evora.core.workspace import Workspace

_SAFE = re.compile(r"^[A-Za-z0-9_:\-]+$")


def similar_tracks(track_id: str, k: int = 20, workspace: Workspace | None = None, *, store=None) -> list[tuple[str, float]]:
    """(track_id, cosine similarity) of the k nearest other tracks of the same class, best first."""
    if not _SAFE.match(track_id):
        raise ValueError(f"invalid track id: {track_id!r}")
    if store is None:
        from evora.perception.pipeline import resolve_workspace

        store = open_store((workspace or resolve_workspace()).vectors_dir)
    if "reid" not in set(store.list_tables().tables):
        return []
    table = store.open_table("reid")
    rows = table.search().where(f"track_id = '{track_id}'").limit(1).to_arrow().to_pylist()
    if not rows:
        return []
    cls = rows[0]["cls"]
    hits = (
        table.search(rows[0]["vector"]).metric("cosine")
        .where(f"cls = '{cls}' AND track_id != '{track_id}'").limit(k).to_arrow().to_pylist()
    )
    return [(h["track_id"], float(1.0 - h["_distance"])) for h in hits]
