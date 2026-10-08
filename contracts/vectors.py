"""LanceDB table specs (contract v1). One workspace owns one `vectors/` directory.

All embeddings are L2-normalized. The vector dimension is read from the workspace `meta`
table (`embed_dim_image`, `embed_dim_reid`, `embed_dim_text`); never mix model versions
inside one table.
"""
from __future__ import annotations

from dataclasses import dataclass

TEXT_MODEL = "BAAI/bge-small-en-v1.5"
TEXT_DIM = 384


@dataclass(frozen=True)
class VectorTable:
    name: str
    vector_model: str          # which embedding model fills the `vector` column
    dim_meta_key: str          # meta key that stores the dimension
    columns: dict[str, str]    # column name -> type: str | float | nullable-str
    writer: str                # owning member


TABLES: dict[str, VectorTable] = {
    "crops": VectorTable(
        "crops", "siglip2-image", "embed_dim_image",
        {"track_id": "str", "camera_id": "str", "cls": "str", "t": "float", "quality": "float", "crop_path": "str"},
        "M2",
    ),
    "scenes": VectorTable(
        "scenes", "siglip2-image", "embed_dim_image",
        {"camera_id": "str", "t": "float", "tile": "str", "frame_path": "str"},
        "M2",
    ),
    "reid": VectorTable(
        "reid", "reid", "embed_dim_reid",
        {"track_id": "str", "camera_id": "str", "cls": "str", "t_start": "float", "t_end": "float"},
        "M2",
    ),
    "captions": VectorTable(
        "captions", TEXT_MODEL, "embed_dim_text",
        {"text": "str", "camera_id": "str", "t": "float", "track_id": "str?"},
        "M2",
    ),
    "aliases": VectorTable(
        "aliases", TEXT_MODEL, "embed_dim_text",
        {"fact_id": "str", "alias": "str"},
        "M1",
    ),
}

TILES = ("full", "tl", "tr", "bl", "br")
