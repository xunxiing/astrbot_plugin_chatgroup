from __future__ import annotations

import os
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from datetime import datetime

from dateutil import parser as dateparser
from bs4 import BeautifulSoup
import emoji


# ------------------------------
# Data structure
# ------------------------------
@dataclass
class Message:
    id: str
    timestamp: Optional[datetime]
    user_id: Optional[str]
    user_name: Optional[str]
    text: str
    reply_to: Optional[str]
    channel: Optional[str]
    attachments: Optional[List[Dict[str, Any]]]
    html: Optional[str]


# ------------------------------
# IO helpers
# ------------------------------
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                data.append(json.loads(s))
            except json.JSONDecodeError:
                # skip malformed line
                pass
    return data


def read_messages(input_path: str) -> List[Message]:
    items: List[Dict[str, Any]] = []
    if os.path.isdir(input_path):
        for fn in sorted(os.listdir(input_path)):
            if fn.lower().endswith(".jsonl"):
                items.extend(load_jsonl(os.path.join(input_path, fn)))
    else:
        items = load_jsonl(input_path)

    messages: List[Message] = []
    for it in items:
        ts: Optional[datetime] = None
        if it.get("timestamp"):
            try:
                ts = dateparser.parse(it["timestamp"])  # tz-aware ok
            except Exception:
                ts = None
        messages.append(
            Message(
                id=str(it.get("id", "")),
                timestamp=ts,
                user_id=it.get("user_id"),
                user_name=it.get("user_name"),
                text=it.get("text", ""),
                reply_to=it.get("reply_to"),
                channel=it.get("channel"),
                attachments=it.get("attachments"),
                html=it.get("html"),
            )
        )
    return messages


# ------------------------------
# Cleaning helpers
# ------------------------------
_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")


def strip_html(html: str) -> str:
    try:
        soup = BeautifulSoup(html, "html.parser")
        txt = soup.get_text(separator=" ")
    except Exception:
        txt = _TAG_RE.sub(" ", html)
    txt = _WHITESPACE_RE.sub(" ", txt).strip()
    return txt


def demojize_text(s: str) -> str:
    try:
        return emoji.demojize(s, language="zh")
    except Exception:
        return s


PLATFORM_PATTERNS = [
    r"\[CQ:[^\]]+\]",     # KOOK/QQ-like inline tags
    r"@\S+",               # mentions
    r"https?://\S+",       # urls
    r"[A-Za-z0-9_]{10,}",  # long ids/hashes
]
PLATFORM_RE = re.compile("|".join(f"(?:{p})" for p in PLATFORM_PATTERNS))


def strip_platform_artifacts(s: str) -> str:
    s = PLATFORM_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


LOW_CONTENT_TOKENS = set(
    [
        "嗯", "啊", "哦", "哈", "呵", "…", "？", "?", "~", "!",
        "可以", "不行", "是的", "好", "好的", "ok", "OK", "已处理", "收到",
    ]
)


def is_low_content(s: str) -> bool:
    if not s or len(s) < 3:
        return True
    if re.fullmatch(r"(?::[a-z0-9_]+:|\W){1,7}", s, flags=re.I):
        return True
    toks = [t for t in re.split(r"\s+", s) if t]
    if 0 < len(toks) <= 3 and all(t in LOW_CONTENT_TOKENS for t in toks):
        return True
    return False


ANCHOR_PAT = re.compile(r"(https?://\S+|\[附件:[^\]]+\]|`{3,}|\?|#\w+)", re.I)


def has_anchor(s: str) -> bool:
    return bool(ANCHOR_PAT.search(s))


def build_message_text(m: Message) -> str:
    base = m.text or ""
    if m.html:
        base = strip_html(m.html)

    base = demojize_text(base)

    # attachments summary
    att_txts: List[str] = []
    if m.attachments:
        for a in m.attachments:
            name = (a.get("title") or a.get("name") or a.get("type") or "").strip()
            stem, _ext = os.path.splitext(name)
            if not stem or len(stem.strip()) < 2:
                continue
            att_txts.append(f"[附件:{stem}]")
    if att_txts:
        base = f"{base} {' '.join(att_txts)}".strip()

    if m.reply_to:
        base = f"{base} [回复:{m.reply_to}]"

    base = strip_platform_artifacts(base)
    if re.fullmatch(r"(?::[a-z0-9_]+:|\W){1,}", base, flags=re.I):
        base = ""
    return base.strip()

