from __future__ import annotations

import os
import time
import logging
from typing import Callable, Dict, List

import numpy as np
from tqdm import tqdm


def retry_with_backoff(max_retries: int = 4, initial_delay: float = 1.0, backoff_factor: float = 2.0):
    """Retry decorator for transient API errors."""
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            last_exc: Exception | None = None
            delay = initial_delay
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:  # noqa: BLE001 - surface provider errors
                    last_exc = e
                    if attempt < max_retries:
                        logging.warning(
                            "Embedding API failed, retry %d in %.1fs: %s",
                            attempt + 1,
                            delay,
                            str(e),
                        )
                        time.sleep(delay)
                        delay *= backoff_factor
                    else:
                        logging.error("Embedding API exhausted retries: %s", str(e))
            assert last_exc is not None
            raise last_exc
        return wrapper
    return decorator


class Embedder:
    def embed(self, texts: List[str], batch_size: int = 64) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError


class SiliconFlowEmbedder(Embedder):
    """Embed via SiliconFlow's OpenAI-compatible embeddings endpoint."""

    def __init__(self, model: str = "BAAI/bge-m3", base_url: str = "https://api.siliconflow.cn/v1/embeddings"):
        import requests  # lazy import

        self.requests = requests
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = os.environ.get("SILICONFLOW_API_KEY")
        # 不在构造函数中强制校验 API Key，避免在仅加载缓存时误报。
        # 如需实际调用远端服务，会在 embed() 中再次校验。

    @retry_with_backoff(max_retries=4, initial_delay=1.0, backoff_factor=2.0)
    def _make_api_request(self, chunk: List[str], headers: Dict[str, str]):
        payload = {"model": self.model, "input": chunk}
        resp = self.requests.post(self.base_url, headers=headers, json=payload, timeout=60)
        if resp.status_code != 200:
            raise RuntimeError(f"SiliconFlow API error: {resp.status_code} {resp.text}")
        data = resp.json()
        embs = [d["embedding"] for d in data.get("data", [])]
        if len(embs) != len(chunk):
            raise RuntimeError("Embeddings count mismatch")
        return embs

    def embed(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        if not self.api_key:
            # Keep message neutral and provider-focused, not env-string specific
            raise RuntimeError("SiliconFlow API Key 未配置")
        out: List[List[float]] = []
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        for i in tqdm(range(0, len(texts), batch_size), desc="embedding", ncols=100):
            chunk = texts[i : i + batch_size]
            embs = self._make_api_request(chunk, headers)
            out.extend(embs)
        return np.asarray(out, dtype=np.float32)


_ST_MODEL = None
try:  # optional dependency
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover - runtime optional import
    SentenceTransformer = None  # type: ignore


class LocalSTEmbedder(Embedder):
    """Local sentence-transformers embedder as a fallback."""

    def __init__(self, model: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        global _ST_MODEL
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers not installed; pip install sentence-transformers")
        if _ST_MODEL is None:
            _ST_MODEL = SentenceTransformer(model)
        self.model = _ST_MODEL

    def embed(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        embs = self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True)
        return np.asarray(embs, dtype=np.float32)
