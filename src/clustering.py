from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import hdbscan
import warnings
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

# 抑制 sklearn 废弃警告
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

from .data_processing import Message


# ------------------------------
# KMeans with auto-K selection
# ------------------------------
def auto_kmeans(X: np.ndarray, k_min: int = 2, k_max: int = 12, random_state: int = 42) -> Tuple[np.ndarray, KMeans]:
    n = X.shape[0]
    if n < max(3, k_min):
        km = KMeans(n_clusters=1, n_init="auto", random_state=random_state)
        labels = km.fit_predict(X)
        return labels, km

    Xn = normalize(X)
    best_score = -1.0
    best_labels: Optional[np.ndarray] = None
    best_km: Optional[KMeans] = None
    k_max_eff = min(k_max, n - 1) if n > 2 else 1
    for k in range(max(2, k_min), max(3, k_max_eff + 1)):
        km = KMeans(n_clusters=k, n_init="auto", random_state=random_state)
        labels = km.fit_predict(Xn)
        if len(set(labels)) < 2:
            continue
        try:
            score = silhouette_score(Xn, labels, metric="cosine")
        except Exception:
            continue
        if score > best_score:
            best_score = score
            best_labels = labels
            best_km = km
    if best_labels is None or best_km is None:
        best_km = KMeans(n_clusters=2, n_init="auto", random_state=random_state)
        best_labels = best_km.fit_predict(Xn)
    return best_labels, best_km


# ------------------------------
# HDBSCAN and helpers
# ------------------------------
def bucket_indices_by_time(msgs: List[Message], idxs: List[int], window_minutes: int) -> Dict[Tuple[str, int], List[int]]:
    buckets: Dict[Tuple[str, int], List[int]] = {}
    for i in idxs:
        m = msgs[i]
        channel = m.channel or "_global"
        if m.timestamp:
            minutes = int(m.timestamp.timestamp() // 60)
            bid = minutes // max(1, window_minutes)
        else:
            bid = -1
        buckets.setdefault((channel, bid), []).append(i)
    return buckets


def dynamic_min_cluster_size(n: int, env: str) -> int:
    if env != "auto":
        try:
            return int(env)
        except Exception:
            pass
    return int(np.clip(np.sqrt(max(n, 1)) * 1.5, 8, 32))


def hdbscan_cluster(X: np.ndarray, min_cluster_size: int, min_samples: int) -> Tuple[np.ndarray, Any]:
    Xn = normalize(X)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=max(2, int(min_cluster_size)),
        min_samples=max(1, int(min_samples)),
        metric="euclidean",
    )
    labels = clusterer.fit_predict(Xn)
    return labels, clusterer


def compute_centroids(X: np.ndarray, labels: np.ndarray) -> Dict[int, np.ndarray]:
    c2v: Dict[int, np.ndarray] = {}
    for c in sorted(set(labels)):
        if c < 0:
            continue
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            continue
        v = normalize(np.mean(X[idx], axis=0, keepdims=True)).ravel()
        c2v[int(c)] = v
    return c2v


def soft_attach_low_content(
    X: np.ndarray,
    labels: np.ndarray,
    low_idxs: List[int],
    msgs: List[Message],
    attach_sim_thr: float = 0.35,
) -> np.ndarray:
    ATTACH_MARGIN = float(os.getenv("ATTACH_MARGIN", "0.07"))
    ATTACH_MAX_MINUTES = int(os.getenv("ATTACH_MAX_MINUTES", "90"))
    HDBSCAN_PROB_THR = float(os.getenv("HDBSCAN_PROB_THR", "0.45"))

    c2v = compute_centroids(X, labels)
    if not c2v:
        return labels
    C = np.stack(list(c2v.values()), axis=0)
    C = normalize(C)
    cid_list = list(c2v.keys())
    Xn = normalize(X)

    id2idx = {m.id: i for i, m in enumerate(msgs)}
    for i in low_idxs:
        rp = msgs[i].reply_to
        if rp and rp in id2idx:
            j = id2idx[rp]
            if labels[j] >= 0:
                labels[i] = labels[j]
                continue

        v = Xn[i : i + 1]
        sims = (v @ C.T).ravel()
        if np.max(sims) < attach_sim_thr:
            labels[i] = -1
            continue
        sorted_sims = np.sort(sims)[::-1]
        if len(sorted_sims) >= 2 and (sorted_sims[0] - sorted_sims[1]) < ATTACH_MARGIN:
            labels[i] = -1
            continue
        k = int(np.argmax(sims))
        target_cid = cid_list[k]

        if msgs[i].timestamp:
            cluster_times = [
                msgs[j].timestamp
                for j in range(len(labels))
                if labels[j] == target_cid and msgs[j].timestamp
            ]
            if cluster_times:
                median_time = np.median([t.timestamp() for t in cluster_times])
                time_diff = abs(msgs[i].timestamp.timestamp() - median_time) / 60
                if time_diff > ATTACH_MAX_MINUTES:
                    labels[i] = -1
                    continue
        labels[i] = target_cid
    return labels


def knn_label_propagation(
    X: np.ndarray,
    labels: np.ndarray,
    candidate_idxs: List[int],
    msgs: List[Message],
    k: int = 5,
    min_in_cluster: int = 2,
    mean_sim_thr: float = 0.34,
    attach_max_minutes: int = 180,
    cluster_median_ts: Optional[Dict[int, int]] = None,
) -> np.ndarray:
    Xn = normalize(X)
    lbl = labels.copy()
    base_idx = np.where(lbl >= 0)[0]
    if len(base_idx) == 0 or len(candidate_idxs) == 0:
        return lbl
    nn = NearestNeighbors(n_neighbors=min(k, len(base_idx)), metric="cosine")
    nn.fit(Xn[base_idx])
    dists, nbrs = nn.kneighbors(Xn[candidate_idxs], return_distance=True)
    sims = 1.0 - dists
    for row, idx in enumerate(candidate_idxs):
        nbr_ids = base_idx[nbrs[row]]
        nbr_labels = lbl[nbr_ids]
        mask = nbr_labels >= 0
        if mask.sum() == 0:
            continue
        labs, counts = np.unique(nbr_labels[mask], return_counts=True)
        best_lab = int(labs[np.argmax(counts)])
        in_cluster = (nbr_labels[mask] == best_lab)
        mean_sim = float(sims[row][mask][in_cluster].mean())
        if counts.max() >= min_in_cluster and mean_sim >= mean_sim_thr:
            pass_time = True
            if cluster_median_ts and msgs[idx].timestamp:
                cts = cluster_median_ts.get(best_lab)
                if cts is not None:
                    dt = abs(int(msgs[idx].timestamp.timestamp()) - cts) / 60.0
                    pass_time = dt <= float(attach_max_minutes)
            if pass_time:
                lbl[idx] = best_lab
    return lbl


# ------------------------------
# Burst & short phrase detection
# ------------------------------
def is_short_phrase(s: str) -> bool:
    t = re.sub(r"[，。！？.!?:：~～…\s]+", "", s)
    if len(t) <= 8:
        return True
    toks = re.split(r"\s+", s.strip())
    return 0 < len(toks) <= 3


def detect_burst_chitchat(
    texts: List[str],
    msgs: List[Message],
    idxs: List[int],
    window_minutes: int = 30,
    min_users: int = 3,
    min_count: int = 3,
):
    buckets = bucket_indices_by_time(msgs, idxs, window_minutes)
    burst_mask = np.zeros(len(texts), dtype=bool)
    for (_, _), bidx in buckets.items():
        counter: Dict[str, Dict[str, Any]] = {}
        for i in bidx:
            s = texts[i]
            if not s or not is_short_phrase(s):
                continue
            u = msgs[i].user_id or msgs[i].user_name or "?"
            key = s.strip()
            info = counter.setdefault(key, {"users": set(), "idx": []})
            info["users"].add(u)
            info["idx"].append(i)
        for _key, info in counter.items():
            if len(info["users"]) >= min_users and len(info["idx"]) >= min_count:
                for i in info["idx"]:
                    burst_mask[i] = True
    return burst_mask

