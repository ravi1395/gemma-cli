"""Memory subpackage for gemma-cli.

The public surface (``MemoryManager``, ``MemoryRetriever`` etc.) is
exposed via PEP 562 module-level ``__getattr__`` so that:

  * ``from gemma.memory.models import MemoryCategory`` stays cheap —
    importing this package no longer eagerly pulls in
    ``manager``, ``retrieval`` and ``store``, all of which import
    NumPy (~30 MB resident).
  * ``from gemma.memory import MemoryManager`` still works for
    consumers that prefer the flat surface; the heavy submodule is
    loaded only when the attribute is actually accessed.

The ``__all__`` list and ``TYPE_CHECKING`` re-exports below preserve
static-analysis and editor-completion behaviour.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover — type-checker only
    from gemma.memory.condensation import CondensationPipeline
    from gemma.memory.context import ContextAssembler
    from gemma.memory.manager import MemoryManager
    from gemma.memory.models import (
        ConversationTurn,
        MemoryCategory,
        MemoryRecord,
    )
    from gemma.memory.retrieval import MemoryRetriever
    from gemma.memory.store import MemoryStore


# Map of public attribute → (submodule, attribute). Used by
# ``__getattr__`` below to resolve attribute access on demand.
_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "CondensationPipeline": ("gemma.memory.condensation", "CondensationPipeline"),
    "ContextAssembler": ("gemma.memory.context", "ContextAssembler"),
    "ConversationTurn": ("gemma.memory.models", "ConversationTurn"),
    "MemoryCategory": ("gemma.memory.models", "MemoryCategory"),
    "MemoryManager": ("gemma.memory.manager", "MemoryManager"),
    "MemoryRecord": ("gemma.memory.models", "MemoryRecord"),
    "MemoryRetriever": ("gemma.memory.retrieval", "MemoryRetriever"),
    "MemoryStore": ("gemma.memory.store", "MemoryStore"),
}


def __getattr__(name: str):
    """Resolve a public attribute by importing its submodule on demand.

    Cached in ``globals()`` after the first lookup so subsequent access
    is a normal attribute fetch with no import cost.
    """
    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        )
    import importlib

    submodule = importlib.import_module(target[0])
    value = getattr(submodule, target[1])
    globals()[name] = value
    return value


__all__ = sorted(_LAZY_ATTRS)
