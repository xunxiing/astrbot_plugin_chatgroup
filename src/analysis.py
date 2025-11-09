from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


# optional jieba
try:  # pragma: no cover - optional import
    import jieba  # type: ignore
    HAS_JIEBA = True
except Exception:  # pragma: no cover - optional import
    HAS_JIEBA = False


def _jieba_tokenize(s: str) -> List[str]:
    return [w.strip() for w in jieba.cut(s) if w.strip()]


DOMAIN_STOPWORDS = set(
    """
cq at reply http https com www img jpg png pdf doc ppt rar zip
哈哈 哈哈哈 呵呵 呵呵呵 呀 是的 可以 不行 嗯 啊 哦 好 好的 OK ok
呵 是 的
""".split()
)


def build_vectorizer(use_jieba: bool = True) -> TfidfVectorizer:
    if use_jieba and HAS_JIEBA:
        return TfidfVectorizer(
            tokenizer=_jieba_tokenize,
            token_pattern=None,
            lowercase=False,
            ngram_range=(1, 2),
            max_features=50000,
            stop_words=list(DOMAIN_STOPWORDS),
        )
    return TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 4),
        max_features=50000,
    )


def ctfidf_terms_for_clusters(cluster_docs: List[str], topk: int = 8, use_jieba: bool = True) -> List[List[str]]:
    if not cluster_docs:
        return []
    vec = build_vectorizer(use_jieba=use_jieba)
    X = vec.fit_transform(cluster_docs)
    vocab = np.array(vec.get_feature_names_out())
    out: List[List[str]] = []
    for i in range(X.shape[0]):
        row = X.getrow(i)
        scores = row.toarray().ravel()
        idx = np.argsort(-scores)[:topk]
        terms = [t for t in vocab[idx] if t not in DOMAIN_STOPWORDS][:topk]
        out.append(terms)
    return out


def make_topic_name(terms: List[str], fallback: str = "讨论主题") -> str:
    if not terms:
        return fallback
    return " / ".join(terms[:4])


def representative_indices(X: np.ndarray, labels: np.ndarray, topk: int = 5) -> Dict[int, List[int]]:
    reps: Dict[int, List[int]] = {}
    for c in sorted(set(labels)):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            reps[c] = []
            continue
        center = normalize(np.mean(X[idx], axis=0, keepdims=True))
        Xi = normalize(X[idx])
        sims = (Xi @ center.T).ravel()
        top_idx = idx[np.argsort(-sims)[:topk]]
        reps[c] = top_idx.tolist()
    return reps

