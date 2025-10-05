from __future__ import annotations

from typing import Any, Dict, List


class ContextClipper:
    """Trim LLM contexts with size and message limits.

    - Keeps system messages.
    - Keeps the most recent non-system messages up to limits.
    - Enforces an approximate character budget.
    """

    def __init__(self, max_chars: int = 6000, max_messages: int = 30) -> None:
        self.max_chars = int(max_chars)
        self.max_messages = int(max_messages)

    def trim(self, contexts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return trim_contexts(contexts, max_chars=self.max_chars, max_messages=self.max_messages)


def trim_contexts(raw_contexts: List[Dict[str, Any]], max_chars: int = 6000, max_messages: int = 30) -> List[Dict[str, Any]]:
    """Keep at most `max_messages` and cap total characters to ~`max_chars`.

    Preserves order. System messages are always kept if present.
    """
    if not isinstance(raw_contexts, list):
        return raw_contexts  # type: ignore[return-value]

    systems: List[Dict[str, Any]] = [m for m in raw_contexts if m.get("role") == "system"]
    others: List[Dict[str, Any]] = [m for m in raw_contexts if m.get("role") != "system"]

    sliced = others[-int(max_messages):]

    total = 0
    kept: List[Dict[str, Any]] = []
    for m in reversed(sliced):
        c = str(m.get("content", ""))
        total += len(c)
        if total > int(max_chars):
            break
        kept.append(m)
    kept.reverse()

    return systems + kept

