"""LanceDB access: open a workspace's vector store and create the contract v1 tables."""
from __future__ import annotations

from pathlib import Path

import lancedb
import pyarrow as pa
from contracts.vectors import TABLES, TEXT_DIM, VectorTable

from evora.core.db import Database

_ARROW = {"str": pa.string(), "float": pa.float64()}


class DimensionError(ValueError):
    pass


def arrow_schema(spec: VectorTable, dim: int) -> pa.Schema:
    fields = [pa.field("vector", pa.list_(pa.float32(), dim))]
    for name, kind in spec.columns.items():
        nullable = kind.endswith("?")
        fields.append(pa.field(name, _ARROW[kind.rstrip("?")], nullable=nullable))
    return pa.schema(fields)


def dims_from_meta(db: Database) -> dict[str, int]:
    """Embedding dimensions per meta key; text dim is fixed by the text model."""
    out = {"embed_dim_text": TEXT_DIM}
    for key in ("embed_dim_image", "embed_dim_reid"):
        value = db.get_meta(key)
        if value is not None:
            out[key] = int(value)
    return out


def open_store(vectors_dir: Path) -> lancedb.DBConnection:
    vectors_dir.mkdir(parents=True, exist_ok=True)
    return lancedb.connect(str(vectors_dir))


def ensure_tables(store: lancedb.DBConnection, dims: dict[str, int], only: set[str] | None = None) -> list[str]:
    """Create any missing table whose dimension is known. Existing tables must match the stored dimension."""
    created: list[str] = []
    existing = set(store.list_tables().tables)
    for name, spec in TABLES.items():
        if only is not None and name not in only:
            continue
        dim = dims.get(spec.dim_meta_key)
        if dim is None:
            continue
        if name in existing:
            have = store.open_table(name).schema.field("vector").type.list_size
            if have != dim:
                raise DimensionError(f"table {name} has dim {have}, workspace expects {dim}; rebuild the workspace")
            continue
        store.create_table(name, schema=arrow_schema(spec, dim))
        created.append(name)
    return created
