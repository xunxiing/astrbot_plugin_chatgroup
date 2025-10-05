from __future__ import annotations

import re
from typing import Any, Dict, Optional


class SpeakDecider:
    """Lightweight decision module for when/what to respond.

    This is intentionally simple (rule-first).
    You can extend it later to call a small model.
    """

    def __init__(self, rules: Optional[Dict[str, Any]] = None) -> None:
        self.rules = rules or {}
        # Basic patterns: question mark or mentions like @bot or #topic
        self._question_pat = re.compile(r"[?？]", re.U)
        self._mention_pat = re.compile(r"(@\w+|#\w+)", re.U)

    def should_process(self, text: Optional[str]) -> bool:
        """Return True if the message is worth processing.

        Heuristics:
        - If empty/None -> False
        - If contains a question mark or a mention token -> True
        - Fallback: True (be permissive by default)
        """
        if not text:
            return False
        t = text.strip()
        if not t:
            return False
        if self._question_pat.search(t):
            return True
        if self._mention_pat.search(t):
            return True
        # Default permissive
        return True

