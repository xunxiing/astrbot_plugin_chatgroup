from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .data_processing import Message


def attach_continuous_by_typing_speed(
    labels: np.ndarray,
    msgs: List[Message],
    texts: List[str],
    max_typing_speed_cps: float = 10.0,
    enabled: bool = True,
) -> np.ndarray:
    """
    Rule: If two consecutive messages from the same user (same channel)
    are close enough such that len(text)/delta_seconds <= max_typing_speed_cps,
    treat as a continuous utterance and attach the later one to the previous
    message's cluster (only if the previous has a valid cluster >=0 and the
    current is unlabeled: -1 or marked chitchat: -2).

    Args:
        labels: current labels (modified in a copy and returned)
        msgs: message objects aligned with labels
        texts: cleaned texts used for char length
        max_typing_speed_cps: threshold (characters per second)
        enabled: toggle

    Returns: new labels array
    """
    if not enabled or max_typing_speed_cps <= 0:
        return labels

    n = len(labels)
    if n == 0:
        return labels

    # Build (channel, timestamp, idx) list and sort
    indexed: List[Tuple[str, float, int]] = []
    for i, m in enumerate(msgs):
        ch = m.channel or "_global"
        ts = m.timestamp.timestamp() if m.timestamp else float("inf")
        indexed.append((ch, ts, i))
    indexed.sort(key=lambda x: (x[0], x[1], x[2]))

    new_labels = labels.copy()
    # Track last index of positive-labeled message per (channel, user)
    last_pos: Dict[Tuple[str, str], int] = {}

    def user_key(m: Message) -> str:
        return m.user_id or m.user_name or "?"

    for ch, ts, i in indexed:
        m = msgs[i]
        if ts == float("inf"):
            # no timestamp — cannot estimate typing speed
            if new_labels[i] >= 0:
                last_pos[(ch, user_key(m))] = i
            continue

        key = (ch, user_key(m))
        prev_i: Optional[int] = last_pos.get(key)

        # If current has no cluster yet (-1 or -2) and we have a previous
        # positive-labeled message for this user in this channel, try attach
        if new_labels[i] < 0 and prev_i is not None and new_labels[prev_i] >= 0:
            prev_m = msgs[prev_i]
            if prev_m.timestamp and m.timestamp:
                dt = (m.timestamp - prev_m.timestamp).total_seconds()
                if dt > 0:
                    char_len = len(texts[i] or "")
                    cps = char_len / dt if dt > 0 else float("inf")
                    if cps <= max_typing_speed_cps:
                        new_labels[i] = int(new_labels[prev_i])

        # Update last positive-labeled pointer
        if new_labels[i] >= 0:
            last_pos[key] = i

    return new_labels


def fill_user_span_within_cluster(
    labels: np.ndarray,
    msgs: List[Message],
    enabled: bool = True,
) -> np.ndarray:
    """
    For each cluster and each user (per channel), find the earliest and latest
    timestamps of that user's messages that are already assigned to the cluster.
    Any other messages by the same user in the same channel whose timestamps lie
    within [earliest, latest] and are currently unlabeled (-1 or -2) will be
    attached to that cluster.

    This implements: "同一用户在某个讨论组中，从最早到最晚发言时间段内的发言也算作该讨论组"。
    """
    if not enabled:
        return labels

    n = len(labels)
    if n == 0:
        return labels

    new_labels = labels.copy()

    # Build per (channel, cluster, user) time ranges
    spans: Dict[Tuple[str, int, str], Tuple[float, float]] = {}

    def ch_of(i: int) -> str:
        return msgs[i].channel or "_global"

    def user_of(i: int) -> str:
        m = msgs[i]
        return m.user_id or m.user_name or "?"

    # First pass: collect min/max ts for each (ch, cluster, user)
    for i in range(n):
        c = int(new_labels[i])
        if c < 0:
            continue
        m = msgs[i]
        if not m.timestamp:
            continue
        ch = ch_of(i)
        u = user_of(i)
        t = m.timestamp.timestamp()
        key = (ch, c, u)
        if key not in spans:
            spans[key] = (t, t)
        else:
            t0, t1 = spans[key]
            if t < t0:
                t0 = t
            if t > t1:
                t1 = t
            spans[key] = (t0, t1)

    if not spans:
        return new_labels

    # Second pass: fill unlabeled messages that fall into any user's span within a cluster
    for i in range(n):
        if new_labels[i] >= 0:
            continue  # already assigned; do not override
        m = msgs[i]
        if not m.timestamp:
            continue
        ch = ch_of(i)
        u = user_of(i)
        ti = m.timestamp.timestamp()
        # try each cluster span for this (channel, user)
        for (ch2, c, u2), (t0, t1) in spans.items():
            if ch2 != ch or u2 != u:
                continue
            if t0 <= ti <= t1:
                new_labels[i] = int(c)
                break

    return new_labels

