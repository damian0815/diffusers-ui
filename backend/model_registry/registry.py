import json
import uuid
from dataclasses import asdict
from pathlib import Path

from backend.domain import ModelEntry


class ModelNotFoundError(Exception):
    """Raised when a model key does not exist in the registry."""


class ModelRegistry:
    """
    Reads and writes models.json.
    One entry per registered model. Keys are stable UUIDs.
    """

    def __init__(self, path: Path):
        self._path = path
        self._entries: dict[str, ModelEntry] = {}
        self._load()

    def all(self) -> list[ModelEntry]:
        return list(self._entries.values())

    def get(self, key: str) -> ModelEntry | None:
        return self._entries.get(key)

    def add(self, entry: ModelEntry) -> ModelEntry:
        """Add an entry to the registry and persist. Returns the entry."""
        self._entries[entry.key] = entry
        self._save()
        return entry

    def register_new(
        self,
        name: str,
        base: str,
        source: str,
        source_type: str,
        hf_api_key: str | None = None,
    ) -> ModelEntry:
        """Create a new entry with a generated UUID key and register it."""
        entry = ModelEntry(
            key=str(uuid.uuid4()),
            name=name,
            base=base,  # type: ignore[arg-type]
            source=source,
            source_type=source_type,  # type: ignore[arg-type]
            hf_api_key=hf_api_key,
        )
        return self.add(entry)

    def rename(self, key: str, new_name: str) -> ModelEntry:
        """Rename a registered model. Raises ModelNotFoundError if key missing."""
        entry = self._entries.get(key)
        if entry is None:
            raise ModelNotFoundError(f"No model with key {key!r}")
        entry.name = new_name
        self._save()
        return entry

    def delete(self, key: str) -> None:
        """Delete a model from the registry. Raises ModelNotFoundError if key missing."""
        if key not in self._entries:
            raise ModelNotFoundError(f"No model with key {key!r}")
        del self._entries[key]
        self._save()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if self._path.exists():
            raw = json.loads(self._path.read_text())
            self._entries = {k: ModelEntry(**v) for k, v in raw.items()}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps({k: asdict(v) for k, v in self._entries.items()}, indent=2)
        )

