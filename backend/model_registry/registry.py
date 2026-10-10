import json
import uuid
from dataclasses import asdict
from pathlib import Path

from backend.domain import ModelEntry, MergeComponent


class ModelNotFoundError(Exception):
    """Raised when a model key does not exist in the registry."""


class ModelMergeError(ValueError):
    """Raised when a merge specification is invalid.

    For structural errors (missing key, non-sana, nested merge, bad weight,
    too few components) `incompatible` is None and str(exc) is the message.
    For architecture mismatches, `incompatible` maps component key ->
    human-readable reason, so the API can surface it per model row.
    """

    def __init__(self, message: str, incompatible: dict[str, str] | None = None):
        super().__init__(message)
        self.incompatible = incompatible


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
        quantization: str | None = None,
        vae_tiling: bool = False,
        use_compel: bool = True,
    ) -> ModelEntry:
        """Create a new entry with a generated UUID key and register it."""
        entry = ModelEntry(
            key=str(uuid.uuid4()),
            name=name,
            base=base,  # type: ignore[arg-type]
            source=source,
            source_type=source_type,  # type: ignore[arg-type]
            hf_api_key=hf_api_key,
            quantization=quantization,
            vae_tiling=vae_tiling,
            use_compel=use_compel,
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

    def update_quantization(self, key: str, quantization: str | None) -> ModelEntry:
        """Set the quantization level for a registered model."""
        entry = self._entries.get(key)
        if entry is None:
            raise ModelNotFoundError(f"No model with key {key!r}")
        entry.quantization = quantization
        self._save()
        return entry

    def update_vae_tiling(self, key: str, enabled: bool) -> ModelEntry:
        """Enable or disable VAE tiling for a registered model."""
        entry = self._entries.get(key)
        if entry is None:
            raise ModelNotFoundError(f"No model with key {key!r}")
        entry.vae_tiling = bool(enabled)
        self._save()
        return entry

    def update_use_compel(self, key: str, enabled: bool) -> ModelEntry:
        """Enable or disable Compel prompt encoding for a registered model."""
        entry = self._entries.get(key)
        if entry is None:
            raise ModelNotFoundError(f"No model with key {key!r}")
        entry.use_compel = bool(enabled)
        self._save()
        return entry

    def delete(self, key: str) -> None:
        """Delete a model from the registry. Raises ModelNotFoundError if key missing."""
        if key not in self._entries:
            raise ModelNotFoundError(f"No model with key {key!r}")
        del self._entries[key]
        self._save()

    def resolve_merge_components(self, components: list[tuple[str, float]]) -> list[ModelEntry]:
        """Validate (key, weight) pairs structurally and return the resolved entries.

        Raises ModelMergeError on the first invalid component. No file access.
        """
        if len(components) < 2:
            raise ModelMergeError("A merge needs at least 2 models.")
        resolved: list[ModelEntry] = []
        for key, weight in components:
            comp = self._entries.get(key)
            if comp is None:
                raise ModelMergeError(f"No model with key {key!r}.")
            if comp.merge is not None:
                raise ModelMergeError(f"{comp.name!r} is itself a merge — nested merges are not supported.")
            if comp.base != "sana":
                raise ModelMergeError(f"{comp.name!r} is a {comp.base} model; only SANA models can be merged.")
            if weight <= 0:
                raise ModelMergeError(f"Weights must be positive (got {weight!r}).")
            resolved.append(comp)
        return resolved

    def add_merge(self, name: str, components: list[tuple[str, float]]) -> ModelEntry:
        """Create and register a merge model from (key, weight) pairs.

        The merge's base/source/source_type/hf_api_key are taken from the FIRST
        component (the base model the pipeline is loaded from).
        """
        resolved = self.resolve_merge_components(components)
        first = resolved[0]
        entry = ModelEntry(
            key=str(uuid.uuid4()),
            name=name,
            base="sana",
            source=first.source,
            source_type=first.source_type,
            hf_api_key=first.hf_api_key,
            quantization=None,
            use_compel=True,
            merge=[MergeComponent(key=comp.key, weight=weight) for comp, weight in zip(resolved, (w for _, w in components))],
        )
        return self.add(entry)

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

