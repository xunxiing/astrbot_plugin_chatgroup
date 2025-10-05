from __future__ import annotations

import os
import time
import json
from typing import Any, Dict, List, Tuple

import numpy as np

from .orchestrator import CharNgramHasher


class ContinuePolicy:
    """Continue-discussion policy: decide withdraw/shorten/supplement/continue.

    Pure decision + text shaping. No direct dependency on event or UI types.
    """

    def __init__(self, orchestrator, embedder, log_path: str) -> None:
        self.orchestrator = orchestrator
        self.embedder = embedder
        self.log_path = log_path
        self._pending: Dict[str, float] = {}
        self._last: Dict[str, Dict[str, Any]] = {}

    # ---- Pending control ----
    def mark_pending(self, chat_id: str) -> None:
        self._pending[chat_id] = time.time()
        self._log(f"[TEST-LOG][continue] mark_pending chat_id={chat_id} ts={self._pending[chat_id]}")

    def should_handle(self, chat_id: str, window: float | None = None) -> bool:
        win = float(os.getenv("CHATGROUP_CONT_HANDLE_WINDOW", str(window or 30.0)))
        ts = self._pending.get(chat_id)
        return bool(ts and (time.time() - ts) <= win)

    def clear_pending(self, chat_id: str) -> None:
        self._pending.pop(chat_id, None)

    # ---- Decision ----
    async def decide(self, chat_id: str, candidate_text: str) -> Tuple[str, Dict[str, Any]]:
        """Return (decision, debug). decision in {withdraw, shorten, supplement, continue}"""
        WIN2 = float(os.getenv("CHATGROUP_CONT_WIN2", "2"))
        WIN5 = float(os.getenv("CHATGROUP_CONT_WIN5", "5"))
        SIM_WITHDRAW = float(os.getenv("CHATGROUP_CONT_SIM_WITHDRAW", "0.85"))
        SIM_PARTIAL = float(os.getenv("CHATGROUP_CONT_SIM_PARTIAL", "0.60"))
        LOOKBACK = int(os.getenv("CHATGROUP_CONT_LOOKBACK", "40"))

        now_s = int(time.time())
        items = self.orchestrator.store.latest(chat_id, limit=max(LOOKBACK, 10))
        recents: List[Dict[str, Any]] = []
        for it in items:
            ts = int(it.get("ts") or 0)
            if now_s - ts <= int(WIN5):
                recents.append(it)
        if not recents:
            debug = {"reason": "no_recent", "recent_count": 0}
            self._log(f"[TEST-LOG][continue] decision chat_id={chat_id} -> continue | debug={json.dumps(debug, ensure_ascii=False)}")
            return "continue", debug

        # Prepare candidate embedding
        if self.embedder is None:
            cand_vec = CharNgramHasher().embed_sync([candidate_text])[0]
        else:
            cand_vec = (await self.embedder.embed([candidate_text]))[0]
        cand_vec = np.asarray(cand_vec, dtype=np.float32).ravel()
        cand_vec = cand_vec / (np.linalg.norm(cand_vec) + 1e-6)

        sims: List[Tuple[float, Dict[str, Any]]] = []
        for it in recents:
            text = str(it.get("text") or "").strip()
            if not text:
                continue
            v = np.asarray(it.get("vec") or [], dtype=np.float32).ravel()
            v = v / (np.linalg.norm(v) + 1e-6)
            sim = float(np.dot(cand_vec, v))
            sims.append((sim, {"text": text, "ts": int(it.get("ts") or 0), "sender": it.get("sender")}))
        if not sims:
            debug = {"reason": "no_vec", "recent_count": len(recents)}
            self._log(f"[TEST-LOG][continue] decision chat_id={chat_id} -> continue | debug={json.dumps(debug, ensure_ascii=False)}")
            return "continue", debug

        sims.sort(key=lambda x: x[0], reverse=True)
        top_sim, top_meta = sims[0]
        sims_top3 = [round(s, 3) for s, _ in sims[:3]]
        dt = now_s - int(top_meta.get("ts") or now_s)
        top_text = str(top_meta.get("text") or "")

        if self._is_low_content(top_text):
            debug = {"reason": "other_low_content", "top_sim": top_sim, "dt": dt, "top_text": top_text[:80], "sims_top3": sims_top3, "recent_count": len(recents)}
            self._log(f"[TEST-LOG][continue] decision chat_id={chat_id} -> continue | debug={json.dumps(debug, ensure_ascii=False)}")
            return "continue", debug

        if dt <= WIN2 and top_sim >= SIM_WITHDRAW:
            debug = {"reason": "win2+highsim", "top_sim": top_sim, "dt": dt, "top_text": top_text[:80], "sims_top3": sims_top3, "recent_count": len(recents)}
            self._log(f"[TEST-LOG][continue] decision chat_id={chat_id} -> withdraw | debug={json.dumps(debug, ensure_ascii=False)}")
            return "withdraw", debug
        if top_sim >= SIM_WITHDRAW:
            debug = {"reason": "highsim", "top_sim": top_sim, "dt": dt, "top_text": top_text[:80], "sims_top3": sims_top3, "recent_count": len(recents)}
            self._log(f"[TEST-LOG][continue] decision chat_id={chat_id} -> withdraw | debug={json.dumps(debug, ensure_ascii=False)}")
            return "withdraw", debug
        if top_sim >= SIM_PARTIAL:
            if top_sim >= (SIM_PARTIAL + (SIM_WITHDRAW - SIM_PARTIAL) * 0.5):
                debug = {"reason": "partial-high", "top_sim": top_sim, "dt": dt, "top_text": top_text[:80], "sims_top3": sims_top3, "recent_count": len(recents)}
                self._log(f"[TEST-LOG][continue] decision chat_id={chat_id} -> shorten | debug={json.dumps(debug, ensure_ascii=False)}")
                return "shorten", debug
            else:
                debug = {"reason": "partial-low", "top_sim": top_sim, "dt": dt, "top_text": top_text[:80], "sims_top3": sims_top3, "recent_count": len(recents)}
                self._log(f"[TEST-LOG][continue] decision chat_id={chat_id} -> supplement | debug={json.dumps(debug, ensure_ascii=False)}")
                return "supplement", debug
        debug = {"reason": "lowsim", "top_sim": top_sim, "dt": dt, "top_text": top_text[:80], "sims_top3": sims_top3, "recent_count": len(recents)}
        self._log(f"[TEST-LOG][continue] decision chat_id={chat_id} -> continue | debug={json.dumps(debug, ensure_ascii=False)}")
        return "continue", debug

    # ---- Text shaping ----
    def shorten_text(self, text: str) -> str:
        import re
        t = (text or '').strip()
        if not t:
            return t
        parts = re.split(r"(?<=[。！？.!?])\s+", t)
        first = parts[0].strip() if parts else t
        if len(first) > 80:
            first = first[:80].rstrip() + "…"
        return f"对，这样做就行：{first}"

    def supplement_text(self, text: str) -> str:
        import re
        t = (text or '').strip()
        if not t:
            return t
        parts = re.split(r"(?<=[。！？.!?])\s+", t)
        tail = "".join(parts[1:]).strip() if len(parts) > 1 else t
        if len(tail) < 1:
            tail = t
        if len(tail) > 120:
            tail = tail[:120].rstrip() + "…"
        return f"另外要注意：{tail}"

    # ---- Utilities ----
    def _is_low_content(self, s: str) -> bool:
        s = (s or '').strip()
        if not s or len(s) < 3:
            return True
        import re
        if re.fullmatch(r"(?::[a-z0-9_]+:|\W){1,7}", s, flags=re.I):
            return True
        low_tokens = {"嗯", "啊", "哦", "哈", "呵", "可以", "不行", "是的", "好的", "OK", "ok", "收到"}
        toks = [t for t in re.split(r"\s+", s) if t]
        if 0 < len(toks) <= 3 and all(t in low_tokens for t in toks):
            return True
        return False

    def _log(self, line: str) -> None:
        try:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
            with open(self.log_path, 'a', encoding='utf-8') as f:
                f.write(line.strip() + '\n')
        except Exception:
            pass
