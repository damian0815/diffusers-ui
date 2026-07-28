import re

EMBEDDING_RE = re.compile(r"<([^>]+)>")


def expand_prompt(text: str, loaded: dict[str, int]) -> str:
    """Expand ``<name>`` tokens in *text* using multi-vector counts from *loaded*.

    Each ``<name>`` where ``name`` is a key in *loaded* is replaced by
    ``name.0 name.1 … name.{k-1}`` where ``k = loaded[name]``.
    Unrecognised ``<name>`` tokens are left untouched.
    """

    def _replace(m: re.Match) -> str:
        n = m.group(1)
        k = loaded.get(n)
        if k is None:
            return m.group(0)  # leave unresolved <name> as-is
        return " ".join(f"{n}.{i}" for i in range(k))

    return EMBEDDING_RE.sub(_replace, text)
