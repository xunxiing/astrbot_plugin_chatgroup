from __future__ import annotations

import os
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ------------------------------
# Persistent storage (SQLite)
# ------------------------------


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


class VectorStore:
    """Simple per-chat message embedding store with a hard cap per chat."""

    def __init__(self, base_dir: str, per_chat_cap: int = 400) -> None:
        self.base_dir = base_dir
        _ensure_dir(self.base_dir)
        self.db_path = os.path.join(self.base_dir, "vectors.db")
        self.per_chat_cap = per_chat_cap
        self._lock = threading.RLock()
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  message_id TEXT,
                  chat_id TEXT NOT NULL,
                  ts INTEGER NOT NULL,
                  sender TEXT,
                  text TEXT NOT NULL,
                  provider TEXT,
                  dim INTEGER NOT NULL,
                  vec BLOB NOT NULL
                )
                """
            )
            con.execute("CREATE INDEX IF NOT EXISTS idx_messages_chat_ts ON messages(chat_id, ts)")
            con.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_messages_mid ON messages(message_id)")

    @staticmethod
    def _to_blob(vec: np.ndarray) -> bytes:
        if vec.dtype != np.float32:
            vec = vec.astype(np.float32)
        return vec.tobytes(order="C")

    @staticmethod
    def _from_blob(blob: bytes, dim: int) -> np.ndarray:
        arr = np.frombuffer(blob, dtype=np.float32)
        if dim and arr.size != dim:
            arr = arr[:dim]
        return arr

    def add(self, chat_id: str, text: str, vec: np.ndarray, *, sender: Optional[str], ts: Optional[int], provider: str, message_id: Optional[str] = None) -> None:
        if not isinstance(vec, np.ndarray):
            raise TypeError("vec must be numpy.ndarray")
        if vec.ndim != 1:
            raise ValueError("vec must be 1-D array")
        dim = int(vec.shape[0])
        blob = self._to_blob(vec)
        message_id = message_id or f"m_{int(time.time()*1000)}_{os.getpid()}"
        ts = int(ts or time.time())

        with self._lock, sqlite3.connect(self.db_path) as con:
            con.execute(
                """
                INSERT OR REPLACE INTO messages(message_id, chat_id, ts, sender, text, provider, dim, vec)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (message_id, chat_id, ts, sender, text, provider, dim, blob),
            )
            self._purge_overflow_locked(con, chat_id)

    def _purge_overflow_locked(self, con: sqlite3.Connection, chat_id: str) -> None:
        cur = con.execute("SELECT COUNT(*) FROM messages WHERE chat_id=?", (chat_id,))
        (count,) = cur.fetchone()
        if count > self.per_chat_cap:
            to_delete = count - self.per_chat_cap
            con.execute("CREATE TEMP TABLE IF NOT EXISTS temp_ids (id INTEGER)")
            con.execute("INSERT INTO temp_ids SELECT id FROM messages WHERE chat_id=? ORDER BY ts ASC LIMIT ?", (chat_id, to_delete))
            con.execute("DELETE FROM messages WHERE id IN (SELECT id FROM temp_ids)")
            con.execute("DROP TABLE temp_ids")

    def latest(self, chat_id: str, limit: int = 400) -> List[Dict[str, Any]]:
        with self._lock, sqlite3.connect(self.db_path) as con:
            cur = con.execute(
                "SELECT message_id, ts, sender, text, provider, dim, vec FROM messages WHERE chat_id=? ORDER BY ts DESC LIMIT ?",
                (chat_id, limit),
            )
            rows = cur.fetchall()
        results: List[Dict[str, Any]] = []
        for mid, ts, sender, text, provider, dim, blob in rows:
            vec = self._from_blob(blob, dim)
            # 灏哊umPy鏁扮粍杞崲涓篜ython鍒楄〃锛岄伩鍏嶅簭鍒楀寲闂
            vec_list = vec.tolist() if isinstance(vec, np.ndarray) else vec
            # 纭繚鎵€鏈夊€奸兘鏄疨ython鍘熺敓绫诲瀷
            results.append({
                "message_id": str(mid),
                "ts": int(ts) if ts is not None else None,
                "sender": str(sender) if sender is not None else None,
                "text": str(text),
                "provider": str(provider),
                "vec": list(vec_list),  # 纭繚鏄疨ython鍒楄〃
            })
        return list(reversed(results))


# ------------------------------
# Embedding via AstrBot providers (async)
# ------------------------------


class CharNgramHasher:
    name = "char-ngram-hash"

    def __init__(self, dim: int = 256, ngram_min: int = 2, ngram_max: int = 3):
        self.dim = dim
        self.ngram_min = ngram_min
        self.ngram_max = ngram_max

    def _ngrams(self, text: str) -> List[str]:
        t = text.strip()
        ngrams: List[str] = []
        for n in range(self.ngram_min, self.ngram_max + 1):
            for i in range(0, max(0, len(t) - n + 1)):
                ngrams.append(t[i : i + n])
        return ngrams

    def embed_sync(self, texts: List[str]) -> np.ndarray:
        mat = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for ng in self._ngrams(text):
                h = hash(ng) % self.dim
                mat[row, h] += 1.0
        mat = mat / (mat.sum(axis=1, keepdims=True) + 1e-6)
        norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-6
        return mat / norms


class AstrBotEmbeddingAdapter:
    """Async embedding adapter that uses AstrBot's configured EmbeddingProvider if available.

    Falls back to a lightweight char n-gram hashing if no provider is available or calls fail.
    """

    name = "astrbot"

    def __init__(self, context, prefer_provider_id: Optional[str] = None, prefer_model: Optional[str] = None) -> None:
        self.context = context
        self.prefer_provider_id = prefer_provider_id
        self.prefer_model = prefer_model
        self._provider = None
        self._fallback = CharNgramHasher()

    def _choose_provider(self):
        try:
            providers = self.context.get_all_embedding_providers()
        except Exception:
            providers = []
        if not providers:
            return None
        if self.prefer_provider_id:
            for p in providers:
                try:
                    if p.meta().id == self.prefer_provider_id:
                        return p
                except Exception:
                    continue
        return providers[0]

    async def embed(self, texts: List[str]) -> np.ndarray:
        prov = self._provider or self._choose_provider()
        if prov is not None:
            try:
                if self.prefer_model:
                    try:
                        prov.set_model(self.prefer_model)
                    except Exception:
                        pass
                embs = await prov.get_embeddings(texts)
                arr = np.asarray(embs, dtype=np.float32)
                norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-6
                self.name = f"astrbot:{prov.meta().id}"
                self._provider = prov
                return arr / norms
            except Exception:
                pass
        # Fallback
        self.name = self._fallback.name
        return self._fallback.embed_sync(texts)


# ------------------------------
# Clustering and topic summary
# ------------------------------


def kmeans_cluster(vecs: np.ndarray, k_min: int = 2, k_max: int = 8) -> Tuple[np.ndarray, int, float]:
    n = vecs.shape[0]
    if n < max(4, k_min * 2):
        return np.zeros(n, dtype=int), 1, 0.0

    try:
        from sklearn.cluster import KMeans  # type: ignore
        from sklearn.metrics import silhouette_score  # type: ignore
    except Exception:
        return np.zeros(n, dtype=int), 1, 0.0

    best_score = -1.0
    best_labels: Optional[np.ndarray] = None
    best_k = 1

    for k in range(k_min, min(k_max, n - 1) + 1):
        try:
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels = km.fit_predict(vecs)
            if len(set(labels)) < 2:
                continue
            score = float(silhouette_score(vecs, labels))
            if score > best_score:
                best_score = score
                best_labels = labels
                best_k = k
        except Exception:
            continue

    if best_labels is None:
        return np.zeros(n, dtype=int), 1, 0.0
    return best_labels, best_k, best_score


def top_keywords(texts: List[str], topk: int = 6) -> List[str]:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    except Exception:
        freq: Dict[str, int] = {}
        for t in texts:
            for ch in t:
                if ch.strip():
                    freq[ch] = freq.get(ch, 0) + 1
        # 纭繚杩斿洖鐨勬槸绾疨ython鍒楄〃
        return list([w for w, _ in sorted(freq.items(), key=lambda x: x[1], reverse=True)[:topk]])

    vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 3), min_df=1)
    X = vec.fit_transform(texts)
    scores = np.asarray(X.sum(axis=0)).ravel()
    idxs = np.argsort(-scores)[:topk]
    inv_vocab = {v: k for k, v in vec.vocabulary_.items()}
    # 纭繚杩斿洖鐨勬槸绾疨ython鍒楄〃
    return list([inv_vocab.get(int(i), "") for i in idxs])


def _unify_vecs(items: List[Dict[str, Any]]) -> np.ndarray:
    """Unify embedding lengths so we can stack safely.

    Strategy: choose the most common dimension among items; truncate or zero-pad others.
    """
    if not items:
        return np.zeros((0, 1), dtype=np.float32)
    lens: List[int] = []
    vec_list: List[np.ndarray] = []
    for it in items:
        v = np.asarray(it.get("vec") or [], dtype=np.float32).ravel()
        vec_list.append(v)
        lens.append(int(v.size))
    if not lens:
        return np.zeros((0, 1), dtype=np.float32)
    try:
        from collections import Counter
        target_dim = Counter(lens).most_common(1)[0][0]
    except Exception:
        target_dim = max(lens) if lens else 1

    out = np.zeros((len(vec_list), int(target_dim)), dtype=np.float32)
    for i, v in enumerate(vec_list):
        n = int(v.size)
        if n >= target_dim:
            out[i, :] = v[:target_dim]
        else:
            out[i, :n] = v
    norms = np.linalg.norm(out, axis=1, keepdims=True) + 1e-6
    out = out / norms
    return out

def build_topics(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not items:
        return []
    # 鎻愬彇鍚戦噺骞惰浆鎹负Python鍒楄〃锛岄伩鍏峃umPy鏁扮粍搴忓垪鍖栭棶棰?
    vecs = _unify_vecs(items)
    labels, best_k, score = kmeans_cluster(vecs)
    if best_k <= 1 or score < 0.05:
        labels = np.zeros(len(items), dtype=int)

    # 灏唋abels杞崲涓篜ython鍒楄〃
    labels_list = labels.tolist()

    # Merge highly similar clusters to avoid splitting one topic
    def _merge_similar(lbls: np.ndarray) -> np.ndarray:
        labs = lbls.astype(int)
        uniq = sorted(set(int(x) for x in labs.tolist()))
        if len(uniq) <= 1:
            return labs

        cos_thr = float(os.getenv("CHATGROUP_MERGE_COSINE_THR", "0.82"))
        cos_hard = float(os.getenv("CHATGROUP_MERGE_COSINE_HARD_THR", "0.88"))
        time_min = int(os.getenv("CHATGROUP_MERGE_TIME_MINUTES", "120"))

        Xn = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-6)
        centers: Dict[int, np.ndarray] = {}
        med_ts: Dict[int, Optional[int]] = {}
        for c in uniq:
            idx = np.where(labs == c)[0]
            if len(idx) == 0:
                continue
            ctr = Xn[idx].mean(axis=0)
            n = np.linalg.norm(ctr) + 1e-6
            centers[c] = ctr / n
            ts_vals = [it.get("ts") for k, it in enumerate(items) if labs[k] == c and isinstance(it.get("ts"), (int, float))]
            med_ts[c] = int(np.median(ts_vals)) if ts_vals else None

        parent = {c: c for c in uniq}
        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra == rb:
                return
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

        for i in range(len(uniq)):
            for j in range(i + 1, len(uniq)):
                c1, c2 = uniq[i], uniq[j]
                v1, v2 = centers.get(c1), centers.get(c2)
                if v1 is None or v2 is None:
                    continue
                sim = float(np.dot(v1, v2))
                if sim >= cos_hard:
                    union(c1, c2)
                    continue
                if sim >= cos_thr:
                    t1, t2 = med_ts.get(c1), med_ts.get(c2)
                    if t1 is not None and t2 is not None:
                        if abs(t1 - t2) / 60.0 <= float(time_min):
                            union(c1, c2)
                    else:
                        union(c1, c2)

        mapping: Dict[int, int] = {}
        next_id = 0
        for c in uniq:
            root = find(c)
            if root not in mapping:
                mapping[root] = next_id
                next_id += 1
            mapping[c] = mapping[root]

        out = labs.copy()
        for k in range(len(out)):
            out[k] = mapping[int(out[k])]
        return out

    try:
        labels = _merge_similar(labels)
        labels_list = labels.tolist()
    except Exception:
        pass

    topics: List[Dict[str, Any]] = []
    for c in sorted(set(map(int, labels_list))):
        members = [it for it, lab in zip(items, labels_list) if int(lab) == c]
        texts = [m.get("text", "") for m in members]
        kws = top_keywords(texts, topk=6)
        topic_name = " / ".join(kws[:4]) if kws else f"璇濋#{c}"
        # 纭繚绀轰緥鏁版嵁鏄函Python绫诲瀷锛屼笉鍖呭惈NumPy鏁扮粍
        ex_src = sorted(members, key=lambda m: m.get("ts") or 0, reverse=True)[:3]
        examples = [
            {
                "sender": (e.get("sender") if isinstance(e, dict) else None),
                "text": (e.get("text") if isinstance(e, dict) else ""),
                "ts": (e.get("ts") if isinstance(e, dict) else None),
            }
            for e in ex_src
        ]
        topics.append({
            "cluster_id": int(c),  # 纭繚鏄疨ython int绫诲瀷
            "topic": topic_name,
            "size": int(len(members)),  # 纭繚鏄疨ython int绫诲瀷
            "keywords": list(kws),  # 纭繚鏄疨ython list绫诲瀷
            "examples": list(examples),  # 纭繚鏄疨ython list绫诲瀷
        })
    topics.sort(key=lambda t: t["size"], reverse=True)
    return topics


class Orchestrator:
    """Coordinate embedding, persistence and topic building."""

    def __init__(self, store_dir: str, per_chat_cap: int = 400, embedder: Optional[Any] = None) -> None:
        self.store = VectorStore(store_dir, per_chat_cap=per_chat_cap)
        self.embedder = embedder  # expected to have async embed(texts)->np.ndarray and name

    async def record_message(self, chat_id: str, text: str, *, sender: Optional[str] = None, ts: Optional[int] = None, message_id: Optional[str] = None) -> None:
        if not text or not text.strip():
            return
        if self.embedder is None:
            # No embedder configured; use fallback
            emb = CharNgramHasher().embed_sync([text])[0]
            provider_name = CharNgramHasher.name
        else:
            arr = await self.embedder.embed([text])
            emb = arr[0]
            provider_name = getattr(self.embedder, "name", "astrbot")
        self.store.add(chat_id=chat_id, text=text.strip(), vec=emb, sender=sender, ts=ts or int(time.time()), provider=provider_name, message_id=message_id)

    def list_topics(self, chat_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        items = self.store.latest(chat_id, limit=limit or self.store.per_chat_cap)
        return build_topics(items)
