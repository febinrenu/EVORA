"""Workspaces: one directory per site, holding its SQLite file, vector tables and media."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from evora.core.config import REPO_ROOT, load_config

_SLUG = re.compile(r"[^a-z0-9]+")
ACTIVE_FILE = ".active"


class WorkspaceError(ValueError):
    pass


@dataclass(frozen=True)
class Workspace:
    slug: str
    root: Path

    @property
    def db_path(self) -> Path:
        return self.root / "evora.sqlite"

    @property
    def vectors_dir(self) -> Path:
        return self.root / "vectors"

    @property
    def media_dir(self) -> Path:
        return self.root / "media"

    @property
    def uploads_dir(self) -> Path:
        return self.root / "uploads"

    @property
    def clips_dir(self) -> Path:
        return self.root / "clips"


def slugify(name: str) -> str:
    slug = _SLUG.sub("-", name.strip().lower()).strip("-")
    if not slug:
        raise WorkspaceError("workspace name must contain letters or digits")
    return slug[:48]


def workspaces_root(root: Path | None = None) -> Path:
    if root is not None:
        return root
    return REPO_ROOT / load_config()["workspace"]["root"]


def _open(slug: str, base: Path) -> Workspace:
    ws = Workspace(slug, base / slug)
    if ws.root.resolve().parent != base.resolve():
        raise WorkspaceError("invalid workspace name")
    return ws


def create(name: str, root: Path | None = None) -> Workspace:
    base = workspaces_root(root)
    ws = _open(slugify(name), base)
    for d in (ws.root, ws.vectors_dir, ws.media_dir, ws.uploads_dir, ws.clips_dir):
        d.mkdir(parents=True, exist_ok=True)
    return ws


def get(slug: str, root: Path | None = None) -> Workspace:
    base = workspaces_root(root)
    ws = _open(slugify(slug), base)
    if not ws.root.is_dir():
        raise WorkspaceError(f"unknown workspace: {slug}")
    return ws


def list_all(root: Path | None = None) -> list[Workspace]:
    base = workspaces_root(root)
    if not base.is_dir():
        return []
    return [Workspace(p.name, p) for p in sorted(base.iterdir()) if p.is_dir()]


def activate(slug: str, root: Path | None = None) -> Workspace:
    ws = get(slug, root)
    (workspaces_root(root)).joinpath(ACTIVE_FILE).write_text(ws.slug, encoding="utf-8")
    return ws


def active(root: Path | None = None) -> Workspace | None:
    marker = workspaces_root(root) / ACTIVE_FILE
    if not marker.is_file():
        return None
    try:
        return get(marker.read_text(encoding="utf-8").strip(), root)
    except WorkspaceError:
        return None
