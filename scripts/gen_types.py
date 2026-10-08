"""Export the contract models as one JSON Schema bundle for TypeScript generation (make types)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from contracts import models  # noqa: E402


def _clean(node):
    """Drop per-field titles (they become noisy alias types) and turn tuples into JSON Schema 'items' arrays."""
    if isinstance(node, dict):
        node.pop("title", None) if "properties" not in node else None
        if "prefixItems" in node:
            node["items"] = node.pop("prefixItems")
            node["minItems"] = node["maxItems"] = len(node["items"])
        for k, v in list(node.items()):
            if k == "properties":
                for prop in v.values():
                    prop.pop("title", None)
            _clean(v)
    elif isinstance(node, list):
        for v in node:
            _clean(v)
    return node


def bundle() -> dict:
    classes = [
        v for v in vars(models).values()
        if isinstance(v, type) and issubclass(v, BaseModel) and v is not BaseModel and v.__module__ == models.__name__
    ]
    defs: dict[str, dict] = {}
    for cls in sorted(classes, key=lambda c: c.__name__):
        schema = cls.model_json_schema(ref_template="#/$defs/{model}")
        defs.update(schema.pop("$defs", {}))
        defs[cls.__name__] = schema
    _clean(defs)
    for name, d in defs.items():
        d["title"] = name
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "EvoraContracts",
        "type": "object",
        "properties": {name: {"$ref": f"#/$defs/{name}"} for name in defs},
        "required": list(defs),
        "additionalProperties": False,
        "$defs": defs,
    }


if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.write_text(json.dumps(bundle(), indent=2) + "\n", encoding="utf-8")
