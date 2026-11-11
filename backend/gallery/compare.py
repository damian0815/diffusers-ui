"""Canonical identity hashing for gallery comparison mode.

Pure functions only — no DB, no torch. The hash of the checked "Match by"
fields defines image identity across models (plan COMPARISON_GALLERY_MODE.md
§3.4). Hashing happens at query time, never stored: the metric is
user-configurable (any subset of 7 fields), so no single hash column can
serve it (§3.1).
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

# Canonical field name -> (attribute name on ImageRecord, null-normalization value).
# Normalization rationale (§3.4): legacy rows may have NULLs where newer rows have
# defaults; without this a NULL/"" pair would hash differently despite being the
# same generation parameters.
FIELD_COLUMNS: dict[str, tuple[str, object]] = {
    "prompt": ("prompt", ""),
    "negative_prompt": ("negative_prompt", ""),
    "seed": ("seed", 0),
    "width": ("width", 0),
    "height": ("height", 0),
    "cfg": ("cfg", 0.0),
    "model": ("model_key", ""),
}

# Sorted canonical names — used by the API default and for validation.
ALL_FIELDS: list[str] = sorted(FIELD_COLUMNS)


def validation_error_fields(fields: list[str]) -> list[str]:
    """Return the sorted set of names in `fields` that are not comparison fields.

    Empty list means every name is valid. The API layer maps a non-empty
    result to HTTP 400; `comparison_hash` raises ValueError with the same list.
    """
    return sorted({f for f in fields if f not in FIELD_COLUMNS})


def _field_value(record_or_dict, attr: str, null_value: object) -> object:
    """Read `attr` from a record object or mapping, normalizing None."""
    if isinstance(record_or_dict, Mapping):
        value = record_or_dict.get(attr)
    else:
        value = getattr(record_or_dict, attr)
    return null_value if value is None else value


def canonical_payload(record_or_dict, fields: list[str]) -> str:
    """JSON string for `record_or_dict` restricted to `fields`, per §3.4.

    Keys are the canonical field names and the payload is serialized with
    sort_keys, so the hash is independent of both the record's attribute
    order and the order of the `fields` list. Raises ValueError on unknown
    field names (the API layer maps this to HTTP 400).
    """
    unknown = validation_error_fields(fields)
    if unknown:
        raise ValueError(f"unknown comparison fields: {', '.join(unknown)}")
    obj = {
        name: _field_value(record_or_dict, *FIELD_COLUMNS[name])
        for name in fields
    }
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def comparison_hash(record, fields: list[str]) -> str:
    """SHA-256 hex digest of `canonical_payload(record, fields)`."""
    payload = canonical_payload(record, fields)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
