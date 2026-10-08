"""SQLite access: schema migration, WAL, and a single serialised writer per database file."""
from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from evora.core.config import REPO_ROOT

SCHEMA_SQL = REPO_ROOT / "contracts" / "schema.sql"
SCHEMA_VERSION = "1.2"
SCHEMA_MAJOR = SCHEMA_VERSION.split(".")[0]

_writers: dict[Path, Database] = {}
_registry_lock = threading.Lock()


class Database:
    """One writer connection (guarded by a lock) plus short-lived read connections."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.RLock()
        self._conn = self._connect()
        self._migrate()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def _migrate(self) -> None:
        with self._lock:
            self._conn.executescript(SCHEMA_SQL.read_text(encoding="utf-8"))
            row = self._conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
            if row is None:
                self._conn.execute("INSERT INTO meta(key, value) VALUES('schema_version', ?)", (SCHEMA_VERSION,))
            elif row["value"].split(".")[0] != SCHEMA_MAJOR:
                raise RuntimeError(f"workspace schema {row['value']} is not compatible with v{SCHEMA_VERSION}")
            elif row["value"] != SCHEMA_VERSION:  # same major: additive tables were just created above
                self._conn.execute("UPDATE meta SET value=? WHERE key='schema_version'", (SCHEMA_VERSION,))

    @contextmanager
    def write(self) -> Iterator[sqlite3.Connection]:
        """Run a block in one transaction on the writer connection; rolls back on error."""
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            self._conn.execute("COMMIT")

    @contextmanager
    def read(self) -> Iterator[sqlite3.Connection]:
        conn = self._connect()
        try:
            yield conn
        finally:
            conn.close()

    def get_meta(self, key: str, default: str | None = None) -> str | None:
        with self.read() as conn:
            row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row["value"] if row else default

    def set_meta(self, key: str, value: str) -> None:
        with self.write() as conn:
            conn.execute(
                "INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def open_db(path: Path) -> Database:
    """Return the process-wide Database for this file, creating and migrating it on first use."""
    key = path.resolve()
    with _registry_lock:
        db = _writers.get(key)
        if db is None:
            db = _writers[key] = Database(key)
        return db


def close_all() -> None:
    with _registry_lock:
        for db in _writers.values():
            db.close()
        _writers.clear()
