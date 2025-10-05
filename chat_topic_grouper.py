#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chat Topic Grouper (文本聊天主题分组)
====================================================

一键把聊天记录分成若干主题组，并给每个组自动起“主题名”。

✅ 特点
- 支持两种向量提供方式：
  1) 本地模型（sentence-transformers）：备用；
  2) **硅基流动 SiliconFlow Embeddings**：默认推荐；模型 `BAAI/bge-m3`（4096维）。
- 自动选择 KMeans 聚类簇数（基于 silhouette score）。
- 以 TF‑IDF（可选 jieba 分词，若无则用中文字符 n‑gram）为每个簇生成关键词，拼成主题名。
- 导出两个文件：
  - `messages_with_labels.jsonl`：每条消息的聚类标签
  - `clusters.json`：每个主题组的关键信息（名称、关键词、代表消息等）

📦 依赖安装（任选其一或都装）：
  pip install scikit-learn emoji tqdm pandas numpy python-dateutil beautifulsoup4
  # 如使用本地向量：
  pip install sentence-transformers
  # 如使用 jieba（更好的中文分词）：
  pip install jieba

🔑 若使用硅基流动：
  - 需设置环境变量：SILICONFLOW_API_KEY
  - 默认使用 base_url: https://api.siliconflow.cn/v1/embeddings

💾 输入格式（JSONL，每行一条消息，示例）：
{
  "id": "m_001",
  "timestamp": "2025-09-20T13:45:12+08:00",
  "user_id": "u_123",
  "user_name": "Alice",
  "text": "午饭点什么？😂",
  "reply_to": null,                    # 或某条消息 id
  "channel": "dev-group",            # 可选
  "attachments": [                    # 可选
    {"type": "file", "name": "设计稿.pdf", "title": "新版导航"}
  ],
  "html": null                        # 若原始是富文本，保留原 HTML 以便清洗
}

▶ 运行：
  python chat_topic_grouper.py \
    --input ./chat.jsonl \
    --provider siliconflow \
    --model "BAAI/bge-m3" \
    --output ./out

  # 使用本地 ST 模型备用：
  python chat_topic_grouper.py --input ./chat.jsonl --provider local --model paraphrase-multilingual-MiniLM-L12-v2

注意：本脚本重点在“能跑 + 工程清晰”。后续你要加图片/OCR/多模态，可在 build_message_text 里把图片/附件摘要拼进文本即可。
"""
from __future__ import annotations

import os
import json
import math
import argparse
import logging
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Callable

from dotenv import load_dotenv
load_dotenv()  # 自动读取 .env 文件

import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from dateutil import parser as dateparser
from bs4 import BeautifulSoup
import re
import hdbscan
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from dotenv import load_dotenv

# 可选：jieba 分词（更适合中文）；若无则退化到 char ngram
try:
    import jieba  # type: ignore
    HAS_JIEBA = True
except Exception:
    HAS_JIEBA = False

# emoji 处理（把 😂 → :face_with_tears_of_joy:）
import emoji

# 本地备选：sentence-transformers（若选择 provider=local 使用）
_ST_MODEL = None
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:
    SentenceTransformer = None  # noqa: N816

# ------------------------------
# 数据结构
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
# 工具函数：IO & 预处理
# ------------------------------

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                logging.warning("跳过无法解析的行: %s | 错误: %s", line[:120], e)
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
        ts = None
        if it.get("timestamp"):
            try:
                ts = dateparser.parse(it["timestamp"])  # tz-aware ok
            except Exception:
                ts = None
        msg = Message(
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
        messages.append(msg)
    return messages


_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")


def strip_html(html: str) -> str:
    # 更稳妥：BeautifulSoup；若失败退回正则
    try:
        soup = BeautifulSoup(html, "html.parser")
        txt = soup.get_text(separator=" ")
    except Exception:
        txt = _TAG_RE.sub(" ", html)
    txt = _WHITESPACE_RE.sub(" ", txt).strip()
    return txt


def demojize_text(s: str) -> str:
    # 😂 → :face_with_tears_of_joy:
    try:
        return emoji.demojize(s, language='zh')
    except Exception:
        return s

# 平台/口水词等噪声清洗
PLATFORM_PATTERNS = [
    r"\[CQ:[^\]]+\]",     # KOOK/QQ 风格内嵌标签
    r"@\S+",              # @提及
    r"https?://\S+",      # URL
    r"[A-Za-z0-9_]{10,}", # 长ID/哈希
]
PLATFORM_RE = re.compile("|".join(f"(?:{p})" for p in PLATFORM_PATTERNS))

LOW_CONTENT_TOKENS = set(list("嗯啊哦哈呵…？?~!") + ["好", "行", "可以", "不行", "是的", "ok", "OK", "已处理", "收到"])

def strip_platform_artifacts(s: str) -> str:
    s = PLATFORM_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def is_low_content(s: str) -> bool:
    # 仅表情/很短/全是口水词 → 低信息
    if not s or len(s) < 3:
        return True
    # 大量 demojize 后的 :xxx: 也视为低信息
    if re.fullmatch(r"(?::[a-z0-9_]+:|\W){1,7}", s, flags=re.I):
        return True
    # 只由少量口水词构成
    toks = [t for t in re.split(r"\s+", s) if t]
    if 0 < len(toks) <= 3 and all(t in LOW_CONTENT_TOKENS for t in toks):
        return True
    return False


ANCHOR_PAT = re.compile(r"(https?://\S+|\\[附件:[^\\]]+\\]|`{3,}|\\?|#\\w+)", re.I)
def has_anchor(s: str) -> bool:
    # 链接、附件占位、代码块、问号、#话题 —— 任一命中即认为有"锚点"
    return bool(ANCHOR_PAT.search(s))


def build_message_text(m: Message) -> str:
    # 优先用 html 清洗，否则用 text
    base = m.text or ""
    if m.html:
        base = strip_html(m.html)

    base = demojize_text(base)

    # 附件提示：只加简要上下文，不要太多噪声
    att_txts = []
    if m.attachments:
        for a in m.attachments:
            name = (a.get("title") or a.get("name") or a.get("type") or "").strip()
            # 忽略只有扩展名或 1~2 个字符的"空名"附件，例如 ".png"
            stem, ext = os.path.splitext(name)
            if not stem or len(stem.strip()) < 2:
                continue
            att_txts.append(f"[附件:{stem}]")
    if att_txts:
        base = f"{base} {' '.join(att_txts)}".strip()

    # 引用/线程提示（若有 reply_to，可加轻量提示；实际 thread 解缠以后再做）
    if m.reply_to:
        base = f"{base} [回复:{m.reply_to}]"
    
    base = strip_platform_artifacts(base)
    # 去掉只由表情/标点组成的占位
    if re.fullmatch(r"(?::[a-z0-9_]+:|\W){1,}", base, flags=re.I):
        base = ""
    return base.strip()


# ------------------------------
# 重试工具函数
# ------------------------------
def retry_with_backoff(max_retries: int = 4, initial_delay: float = 1.0, backoff_factor: float = 2.0):
    """
    重试装饰器，在函数抛出异常时自动重试
    
    Args:
        max_retries: 最大重试次数
        initial_delay: 初始延迟时间（秒）
        backoff_factor: 延迟时间增长因子
    """
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            last_exception = None
            delay = initial_delay
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        logging.warning(f"API 调用失败，第 {attempt + 1} 次重试，延迟 {delay:.1f} 秒。错误: {str(e)}")
                        time.sleep(delay)
                        delay *= backoff_factor
                    else:
                        logging.error(f"API 调用失败，已达最大重试次数 {max_retries}。错误: {str(e)}")
            
            # 如果所有重试都失败了，抛出最后一个异常
            raise last_exception
        return wrapper
    return decorator


# ------------------------------
# 向量提供者（Provider）
# ------------------------------
class Embedder:
    def embed(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        raise NotImplementedError


class SiliconFlowEmbedder(Embedder):
    """使用硅基流动 OpenAI‑兼容 Embeddings API。

    需要环境变量：SILICONFLOW_API_KEY
    默认 endpoint: https://api.siliconflow.cn/v1/embeddings
    模型：BAAI/bge-m3（4096 维）
    """
    def __init__(self, model: str = "BAAI/bge-m3", base_url: str = "https://api.siliconflow.cn/v1/embeddings"):
        import requests  # 延迟导入
        self.requests = requests
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = os.environ.get("SILICONFLOW_API_KEY")
        if not self.api_key:
            raise RuntimeError("未找到环境变量 SILICONFLOW_API_KEY。请先 export SILICONFLOW_API_KEY=...")

    @retry_with_backoff(max_retries=4, initial_delay=1.0, backoff_factor=2.0)
    def _make_api_request(self, chunk: List[str], headers: Dict[str, str]):
        """内部方法：发送API请求，带有重试功能"""
        payload = {
            "model": self.model,
            "input": chunk,
        }
        resp = self.requests.post(self.base_url, headers=headers, json=payload, timeout=60)
        if resp.status_code != 200:
            raise RuntimeError(f"SiliconFlow API 错误: {resp.status_code} {resp.text}")
        data = resp.json()
        # 兼容 OpenAI embeddings 返回结构
        embs = [d["embedding"] for d in data.get("data", [])]
        if len(embs) != len(chunk):
            raise RuntimeError("返回的 embedding 数量与输入不一致")
        return embs

    def embed(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        out: List[List[float]] = []
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        for i in tqdm(range(0, len(texts), batch_size), desc="embedding", ncols=100):
            chunk = texts[i:i + batch_size]
            embs = self._make_api_request(chunk, headers)
            out.extend(embs)
        return np.asarray(out, dtype=np.float32)


class LocalSTEmbedder(Embedder):
    """本地 sentence-transformers 作为备用方案。"""
    def __init__(self, model: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        global _ST_MODEL
        if SentenceTransformer is None:
            raise RuntimeError("未安装 sentence-transformers，请 pip install sentence-transformers")
        if _ST_MODEL is None:
            _ST_MODEL = SentenceTransformer(model)
        self.model = _ST_MODEL

    def embed(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        embs = self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True)
        return np.asarray(embs, dtype=np.float32)


# ------------------------------
# 聚类（KMeans + 自动选 K）
# ------------------------------
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize


def auto_kmeans(X: np.ndarray, k_min: int = 2, k_max: int = 12, random_state: int = 42) -> Tuple[np.ndarray, KMeans]:
    n = X.shape[0]
    if n < max(3, k_min):
        # 样本太少，全部归为一类
        km = KMeans(n_clusters=1, n_init="auto", random_state=random_state)
        labels = km.fit_predict(X)
        return labels, km

    # 归一化余弦空间（更稳）
    Xn = normalize(X)

    best_k, best_score, best_km, best_labels = None, -1.0, None, None
    k_max = min(k_max, n - 1) if n > 2 else 1
    for k in range(max(2, k_min), max(3, k_max + 1)):
        km = KMeans(n_clusters=k, n_init="auto", random_state=random_state)
        labels = km.fit_predict(Xn)
        # 某些聚类可能产生单一簇，跳过
        if len(set(labels)) < 2:
            continue
        try:
            score = silhouette_score(Xn, labels, metric="cosine")
        except Exception:
            continue
        if score > best_score:
            best_k, best_score, best_km, best_labels = k, score, km, labels

    if best_labels is None:
        # 兜底：强制 2 类
        best_km = KMeans(n_clusters=2, n_init="auto", random_state=random_state)
        best_labels = best_km.fit_predict(Xn)

    return best_labels, best_km  # type: ignore


# ------------------------------
# HDBSCAN 聚类、时间分桶与软吸附
# ------------------------------
def is_short_phrase(s: str) -> bool:
    # 清轻度符号，只要 8 字以内或 <=3 个分词，就视为短语
    t = re.sub(r"[，。！？,.!?:：~～…\s]+", "", s)
    if len(t) <= 8:
        return True
    toks = re.split(r"\s+", s.strip())
    return 0 < len(toks) <= 3

def detect_burst_chitchat(texts, msgs, idxs, window_minutes=30, min_users=3, min_count=3):
    buckets = bucket_indices_by_time(msgs, idxs, window_minutes)
    burst_mask = np.zeros(len(texts), dtype=bool)
    for (_, _), bidx in buckets.items():
        # 统计"短语" -> {user_id set, indices}
        counter = {}
        for i in bidx:
            s = texts[i]
            if not s or not is_short_phrase(s):
                continue
            u = msgs[i].user_id or msgs[i].user_name or "?"
            key = s.strip()
            info = counter.setdefault(key, {"users": set(), "idx": []})
            info["users"].add(u)
            info["idx"].append(i)
        for key, info in counter.items():
            if len(info["users"]) >= min_users and len(info["idx"]) >= min_count:
                # 打标签为爆发式闲聊
                for i in info["idx"]:
                    burst_mask[i] = True
    return burst_mask

def bucket_indices_by_time(msgs: List[Message], idxs: List[int], window_minutes: int) -> Dict[Tuple[str, int], List[int]]:
    """
    按 channel + 时间窗口分桶；无 timestamp 统一放 -1。
    返回: {(channel, bucket_id): [indices...]}
    """
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
    # sqrt 策略：样本越多阈值越大；最小 8，最大 32（可按需调）
    return int(np.clip(np.sqrt(max(n,1))*1.5, 8, 32))

def hdbscan_cluster(X: np.ndarray, min_cluster_size: int, min_samples: int) -> Tuple[np.ndarray, Any]:
    Xn = normalize(X)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=max(2, int(min_cluster_size)),
        min_samples=max(1, int(min_samples)),
        metric="euclidean"  # 在 L2 归一化后等价于 cosine
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
    """
    将低信息消息/噪声按簇心相似度吸附；若 reply_to 指向已分簇的消息，则优先跟随。
    添加三个门槛：概率阈值、相似度落差、时间门槛。
    """
    # 读取参数
    ATTACH_MARGIN = float(os.getenv("ATTACH_MARGIN", "0.07"))
    ATTACH_MAX_MINUTES = int(os.getenv("ATTACH_MAX_MINUTES", "90"))
    HDBSCAN_PROB_THR = float(os.getenv("HDBSCAN_PROB_THR", "0.45"))
    
    # 先把所有非负簇的质心算好
    c2v = compute_centroids(X, labels)
    if not c2v:
        return labels
    C = np.stack(list(c2v.values()), axis=0)  # [C, D]
    C = normalize(C)
    cid_list = list(c2v.keys())
    Xn = normalize(X)

    # 建 id -> 索引映射，便于查 reply_to
    id2idx = {m.id: i for i, m in enumerate(msgs)}
    for i in low_idxs:
        # 若 reply_to 已有簇，直接跟随
        rp = msgs[i].reply_to
        if rp and rp in id2idx:
            j = id2idx[rp]
            if labels[j] >= 0:
                labels[i] = labels[j]
                continue
        
        # 否则看质心相似度
        v = Xn[i:i+1]          # [1, D]
        sims = (v @ C.T).ravel()
        
        # 如果没有足够相似的簇，跳过
        if np.max(sims) < attach_sim_thr:
            labels[i] = -1
            continue
            
        # 检查相似度落差（Top1 - Top2 >= ATTACH_MARGIN）
        sorted_sims = np.sort(sims)[::-1]
        if len(sorted_sims) >= 2 and (sorted_sims[0] - sorted_sims[1]) < ATTACH_MARGIN:
            labels[i] = -1
            continue
            
        # 找到最相似的簇
        k = int(np.argmax(sims))
        target_cid = cid_list[k]
        
        # 检查时间门槛（消息时间与簇中位时间差 <= ATTACH_MAX_MINUTES）
        if msgs[i].timestamp:
            # 计算目标簇的时间中位数
            cluster_times = []
            for j in range(len(labels)):
                if labels[j] == target_cid and msgs[j].timestamp:
                    cluster_times.append(msgs[j].timestamp)
            
            if cluster_times:
                median_time = np.median([t.timestamp() for t in cluster_times])
                time_diff = abs(msgs[i].timestamp.timestamp() - median_time) / 60  # 转换为分钟
                if time_diff > ATTACH_MAX_MINUTES:
                    labels[i] = -1
                    continue
        
        # 所有门槛都通过，分配到该簇
        labels[i] = target_cid
        
    return labels


from sklearn.neighbors import NearestNeighbors

def knn_label_propagation(
    X: np.ndarray,
    labels: np.ndarray,
    candidate_idxs: List[int],
    msgs: List[Message],
    k: int = 5,
    min_in_cluster: int = 2,
    mean_sim_thr: float = 0.34,
    attach_max_minutes: int = 180,
    cluster_median_ts: Optional[Dict[int,int]] = None,
) -> np.ndarray:
    Xn = normalize(X)
    lbl = labels.copy()
    # 已有簇的样本作为"图库"
    base_idx = np.where(lbl >= 0)[0]
    if len(base_idx) == 0 or len(candidate_idxs) == 0:
        return lbl
    nn = NearestNeighbors(n_neighbors=min(k, len(base_idx)), metric="cosine")
    nn.fit(Xn[base_idx])
    dists, nbrs = nn.kneighbors(Xn[candidate_idxs], return_distance=True)
    # cosine 距离 -> 相似度
    sims = 1.0 - dists
    for row, idx in enumerate(candidate_idxs):
        nbr_ids = base_idx[nbrs[row]]
        nbr_labels = lbl[nbr_ids]
        # 取邻居中已标簇的
        mask = nbr_labels >= 0
        if mask.sum() == 0:
            continue
        # 投票：找出现次数最多的簇，并计算该簇邻居的平均相似度
        labs, counts = np.unique(nbr_labels[mask], return_counts=True)
        best_lab = int(labs[np.argmax(counts)])
        in_cluster = (nbr_labels[mask] == best_lab)
        mean_sim = float(sims[row][mask][in_cluster].mean())
        if counts.max() >= min_in_cluster and mean_sim >= mean_sim_thr:
            # 时间门槛
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
# 主题命名（TF‑IDF 提取关键词）
# ------------------------------
from sklearn.feature_extraction.text import TfidfVectorizer


def _jieba_tokenize(s: str) -> List[str]:
    return [w.strip() for w in jieba.cut(s) if w.strip()]

DOMAIN_STOPWORDS = set("""
cq at reply http https com www img jpg png pdf doc ppt rar zip
哈哈 哈哈哈 嗯 啊 哦 呢 呀 呃 是的 可以 不行 好 行 ？ ?
""".split())

def build_vectorizer(use_jieba: bool = True) -> TfidfVectorizer:
    if use_jieba and HAS_JIEBA:
        return TfidfVectorizer(
            tokenizer=_jieba_tokenize,
            token_pattern=None,
            lowercase=False,
            ngram_range=(1, 2),
            max_features=50000,
            stop_words=list(DOMAIN_STOPWORDS)
        )
    # 退化：中文字符 n-gram（不依赖分词也能凑合）
    return TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 4),
        max_features=50000,
    )

def ctfidf_terms_for_clusters(cluster_docs: List[str], topk: int = 8, use_jieba: bool = True) -> List[List[str]]:
    """
    c-TF-IDF: 把每个簇的文本拼接成"大文档"，再做 TF-IDF。
    返回：每个簇的关键词列表。
    """
    if not cluster_docs:
        return []
    vec = build_vectorizer(use_jieba=use_jieba)
    X = vec.fit_transform(cluster_docs)  # shape [C, V]
    vocab = np.array(vec.get_feature_names_out())
    # 取每个簇文档的 topk 词
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


# ------------------------------
# 代表消息（簇心）
# ------------------------------

def representative_indices(X: np.ndarray, labels: np.ndarray, topk: int = 5) -> Dict[int, List[int]]:
    reps: Dict[int, List[int]] = {}
    for c in sorted(set(labels)):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            reps[c] = []
            continue
        # 用簇内平均向量做“中心”，选余弦相似度最高的前 k 条
        center = normalize(np.mean(X[idx], axis=0, keepdims=True))
        Xi = normalize(X[idx])
        sims = (Xi @ center.T).ravel()
        top_idx = idx[np.argsort(-sims)[:topk]]
        reps[c] = top_idx.tolist()
    return reps


# ------------------------------
# 主流程
# ------------------------------

def run(
    input_path: str,
    provider: str = "siliconflow",
    model: str = "BAAI/bge-m3",
    output_dir: str = "./out",
    k_min: int = 2,
    k_max: int = 12,
    batch_size: int = 64,
    use_jieba: bool = True,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    logging.info("读取输入……")
    msgs = read_messages(input_path)
    if not msgs:
        raise SystemExit("没有读到任何消息。请检查 --input")

    logging.info("构建文本……")
    texts = [build_message_text(m) for m in msgs]

    logging.info("向量计算（provider=%s, model=%s）……", provider, model)
    if provider.lower() == "siliconflow":
        embedder: Embedder = SiliconFlowEmbedder(model=model)
    elif provider.lower() == "local":
        embedder = LocalSTEmbedder(model=model)
    else:
        raise SystemExit("provider 仅支持 siliconflow / local")

    X = embedder.embed(texts, batch_size=batch_size)
    if not np.all(np.isfinite(X)):
        raise RuntimeError("Embedding 中存在 inf/nan，请检查输入或 provider 返回值。")

    # 读取 .env 中的参数（可被 CLI 参数部分覆盖/无影响）
    WINDOW_MINUTES = int(os.getenv("WINDOW_MINUTES", "30"))
    H_MIN_CLUSTER = int(os.getenv("HDBSCAN_MIN_CLUSTER_SIZE", "20"))
    H_MIN_SAMPLES = int(os.getenv("HDBSCAN_MIN_SAMPLES", "5"))
    ATTACH_SIM_THR = float(os.getenv("ATTACH_SIM_THR", "0.35"))
    ATTACH_MARGIN = float(os.getenv("ATTACH_MARGIN", "0.07"))
    ATTACH_MAX_MINUTES = int(os.getenv("ATTACH_MAX_MINUTES", "90"))
    MAX_RECLUSTER_SIZE = int(os.getenv("MAX_RECLUSTER_SIZE", "80"))
    
    # 标记低信息样本（不直接参与聚类）
    low_mask = np.array([is_low_content(t) for t in texts], dtype=bool)
    
    # 计算闲聊中心（用于检测与闲聊相似的内容）
    CHITCHAT_SIM_THR = float(os.getenv("CHITCHAT_SIM_THR", "0.68"))
    # 使用低信息内容的平均向量作为闲聊中心
    low_content_indices = [i for i in range(len(texts)) if low_mask[i]]
    if low_content_indices:
        chit_center = normalize(np.mean(X[low_content_indices], axis=0, keepdims=True)).ravel()
    else:
        chit_center = np.zeros(X.shape[1])
    
    # 检测与闲聊中心相似的内容
    def is_chitchat_by_vector(vec, center, thr):
        if np.all(center == 0):  # 如果没有闲聊中心
            return False
        sim = np.dot(vec, center)
        return sim >= thr
    
    Xn_all = normalize(X)
    chit_mask = np.array([is_chitchat_by_vector(Xn_all[i:i+1], chit_center, CHITCHAT_SIM_THR) for i in range(len(texts))])
    
    # 新增：爆发式短语检测（同桶内多人复读）
    BURST_MIN_USERS = int(os.getenv("BURST_MIN_USERS", "3"))
    BURST_MIN_COUNT = int(os.getenv("BURST_MIN_COUNT", "3"))
    burst_mask = detect_burst_chitchat(texts, msgs, list(range(len(texts))), window_minutes=WINDOW_MINUTES,
                                       min_users=BURST_MIN_USERS, min_count=BURST_MIN_COUNT)
    
    anchor_mask = np.array([has_anchor(t) for t in texts], dtype=bool)
    
    # 核心集合：不是低信息，且（不是闲聊/爆发 或 有锚点）
    core_mask = (~low_mask) & ( (~(chit_mask | burst_mask)) | anchor_mask )
    border_mask = (~low_mask) & (~core_mask)   # 其余非低信息的，后续用 kNN 补回
    
    # 确保数组是一维的
    core_mask = core_mask.flatten()
    border_mask = border_mask.flatten()
    
    valid_idxs  = [i for i in range(len(texts)) if core_mask[i]]
    border_idxs = [i for i in range(len(texts)) if border_mask[i]]
    logging.info("核心: %d | 边界: %d | 低信息: %d",
                 len(valid_idxs), len(border_idxs), int(low_mask.sum()))

    # 1) 时间分桶 + HDBSCAN
    logging.info("按 channel+%d分钟 分桶聚类（HDBSCAN 优先）……", WINDOW_MINUTES)
    global_labels = np.full(len(texts), -1, dtype=int)
    buckets = bucket_indices_by_time(msgs, valid_idxs, WINDOW_MINUTES)
    next_cid = 0
    for (ch, bid), idxs in buckets.items():
        subX = X[idxs]
        mcs = dynamic_min_cluster_size(len(idxs), os.getenv("DYNAMIC_MIN_CLUSTER","auto"))
        if len(idxs) < max(mcs, 10):
            sub_labels, _ = auto_kmeans(subX, k_min=2, k_max=min(6, len(idxs)))
        else:
            sub_labels, sub_clusterer = hdbscan_cluster(subX, mcs, H_MIN_SAMPLES)
            try:
                probs = sub_clusterer.probabilities_
                HDBSCAN_PROB_THR = float(os.getenv("HDBSCAN_PROB_THR", "0.38"))
                sub_labels = np.where(probs >= HDBSCAN_PROB_THR, sub_labels, -1)
            except Exception:
                pass
            # 若噪声占比过大（>60%），说明过严 → 回退 KMeans
            if (sub_labels < 0).mean() > 0.60:
                sub_labels, _ = auto_kmeans(subX, k_min=2, k_max=min(6, len(idxs)))
        # 映射到全局 label
        unique = sorted(set([l for l in sub_labels if l >= 0]))
        mapping = {l: (next_cid + i) for i, l in enumerate(unique)}
        for j, lab in zip(idxs, sub_labels):
            if lab >= 0:
                global_labels[j] = mapping[lab]
        next_cid += len(unique)

    # 2) 递归细分巨无霸簇（简单一轮）
    for c in sorted(set(global_labels)):
        if c < 0:
            continue
        members = np.where(global_labels == c)[0]
        if len(members) > MAX_RECLUSTER_SIZE:
            sub_labels, _ = auto_kmeans(X[members], k_min=2, k_max=min(8, len(members)-1))
            # 重标号
            uniq = sorted(set(sub_labels))
            remap = {l: (next_cid + i) for i, l in enumerate(uniq)}
            for k, lab in zip(members, sub_labels):
                global_labels[k] = remap[lab]
            next_cid += len(uniq)

    # 3) 先对"边界+噪声"用 kNN 标签传播补全，再用质心软吸附兜底
    # 确保 border_mask 与 global_labels 形状一致
    if len(border_mask) != len(global_labels):
        border_mask = border_mask[:len(global_labels)]
    
    # 进阶：明确标注闲聊，把纯闲聊打成固定标签-2
    for i in range(len(texts)):
        if chit_mask[i] and not anchor_mask[i] and not low_mask[i]:
            global_labels[i] = -2  # 标记为闲聊
    # 修改KNN传播的candidate_idxs构造，只对"非低信息、非闲聊"的未归属/边界样本做传播
    cand = [i for i in range(len(texts))
            if ((global_labels[i] < 0) or border_mask[i])
            and (global_labels[i] != -2)      # 跳过已标记的闲聊样本
            and (not low_mask[i])             # 排除低信息
            and (not (chit_mask[i] and not anchor_mask[i]))]  # 排除纯闲聊
    
    # 计算每个簇的时间中位数，用于时间门槛检查
    cluster_median_ts = {}
    for c in sorted(set(global_labels)):
        if c < 0:
            continue
        cluster_times = []
        for i in range(len(global_labels)):
            if global_labels[i] == c and msgs[i].timestamp:
                cluster_times.append(msgs[i].timestamp.timestamp())
        if cluster_times:
            cluster_median_ts[c] = int(np.median(cluster_times))
    
    # 3.1 kNN 标签传播（先补一轮）
    KNN_K = int(os.getenv("KNN_K","5"))
    KNN_MIN_IN_CLUSTER = int(os.getenv("KNN_MIN_IN_CLUSTER","2"))
    KNN_MEAN_SIM_THR = float(os.getenv("KNN_MEAN_SIM_THR","0.34"))
    global_labels = knn_label_propagation(
        X, global_labels, cand, msgs,
        k=KNN_K, min_in_cluster=KNN_MIN_IN_CLUSTER, mean_sim_thr=KNN_MEAN_SIM_THR,
        attach_max_minutes=ATTACH_MAX_MINUTES, cluster_median_ts=cluster_median_ts
    )
    
    # 3.2 仍未归属的再用质心软吸附兜底（更宽松）
    # 软吸附前再过滤，只允许非低信息、非闲聊进入软吸附候选
    remain = [i for i in range(len(texts))
              if global_labels[i] < 0
              and (global_labels[i] != -2)      # 跳过已标记的闲聊样本
              and (not low_mask[i])
              and (not (chit_mask[i] and not anchor_mask[i]))]
    global_labels = soft_attach_low_content(
        X=X, labels=global_labels, low_idxs=remain, msgs=msgs,
        attach_sim_thr=ATTACH_SIM_THR
    )
    
    # 3.3 二次发现：对剩余未归属样本做一次小 HDBSCAN
    unassigned = np.where(global_labels < 0)[0]
    if len(unassigned) >= 12:
        mcs2 = max(6, int(dynamic_min_cluster_size(len(unassigned), "auto") * 0.6))
        sub_labels2, sub_clusterer2 = hdbscan_cluster(X[unassigned], mcs2, max(3, H_MIN_SAMPLES-1))
        # 把 -1 以外的映射到新的全局簇
        uniq2 = sorted(set([l for l in sub_labels2 if l >= 0]))
        mapping2 = {l: (next_cid + i) for i, l in enumerate(uniq2)}
        for u, lab in zip(unassigned, sub_labels2):
            if lab >= 0:
                global_labels[u] = mapping2[lab]
        next_cid += len(uniq2)

    labels = global_labels
    reps = representative_indices(X, labels, topk=5)

    logging.info("主题命名（c-TF-IDF）……")
    df = pd.DataFrame({
        "id": [m.id for m in msgs],
        "timestamp": [m.timestamp.isoformat() if m.timestamp else None for m in msgs],
        "user_id": [m.user_id for m in msgs],
        "user_name": [m.user_name for m in msgs],
        "channel": [m.channel for m in msgs],
        "text": texts,
        "label": labels,
    })

    clusters: List[Dict[str, Any]] = []
    valid_clusters = [c for c in sorted(df["label"].unique()) if c >= 0]
    # 每个簇拼接文档做 c-TF-IDF
    docs = [" ".join(df[df.label == c]["text"].tolist()) for c in valid_clusters]
    terms_list = ctfidf_terms_for_clusters(docs, topk=8, use_jieba=use_jieba)
    c2terms = {c: terms for c, terms in zip(valid_clusters, terms_list)}

    for c in valid_clusters:
        sub = df[df["label"] == c]
        terms = c2terms.get(c, [])
        topic = make_topic_name(terms, fallback=f"主题#{c}")
        rep_rows = sub.loc[reps.get(c, [])]
        clusters.append({
            "cluster_id": int(c),
            "topic": topic,
            "size": int(len(sub)),
            "keywords": terms[:8],
            "representative_messages": [
                {
                    "id": r["id"],
                    "user": r["user_name"],
                    "timestamp": r["timestamp"],
                    "text": r["text"],
                }
                for _, r in rep_rows.iterrows()
            ],
            "message_ids": sub["id"].tolist(),
        })

    # 导出
    out_jsonl = os.path.join(output_dir, "messages_with_labels.jsonl")
    out_clusters = os.path.join(output_dir, "clusters.json")

    logging.info("写出：%s", out_jsonl)
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")

    logging.info("写出：%s", out_clusters)
    with open(out_clusters, "w", encoding="utf-8") as f:
        json.dump(clusters, f, ensure_ascii=False, indent=2)

    total = len(df)
    noise_cnt = int((df["label"] == -1).sum())
    chatter_cnt = int((df["label"] == -2).sum()) if (-2 in df["label"].unique()) else 0
    border_joined = len([i for i in border_idxs if labels[i] >= 0])
    logging.info("总数: %d | 主题簇: %d | 噪声: %d | 闲聊: %d | 边界补回: %d",
                 total, len([c for c in set(labels) if c >= 0]), noise_cnt, chatter_cnt, border_joined)
    
    logging.info("完成。共 %d 条消息，被分成 %d 个主题组。", len(df), len(clusters))


# ------------------------------
# CLI
# ------------------------------

def parse_args() -> argparse.Namespace:
    # 从环境变量读取配置，如果没有则使用默认值
    input_path = os.getenv("INPUT", "./chat.jsonl")
    provider = os.getenv("PROVIDER", "siliconflow")
    model = os.getenv("MODEL", "BAAI/bge-m3")
    output_dir = os.getenv("OUTPUT", "./out")
    k_min = int(os.getenv("K_MIN", "2"))
    k_max = int(os.getenv("K_MAX", "12"))
    batch_size = int(os.getenv("BATCH_SIZE", "64"))
    
    p = argparse.ArgumentParser(
        description="把聊天记录分组为主题（SiliconFlow BAAI/bge-m3 4096维）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", default=input_path, help="输入 .jsonl 文件或包含 .jsonl 的目录")
    p.add_argument("--provider", default=provider, choices=["siliconflow", "local"], help="向量提供者")
    p.add_argument("--model", default=model, help="向量模型名（siliconflow/local）")
    p.add_argument("--output", default=output_dir, help="输出目录")
    p.add_argument("--k_min", type=int, default=k_min, help="KMeans 最小簇数")
    p.add_argument("--k_max", type=int, default=k_max, help="KMeans 最大簇数")
    p.add_argument("--batch_size", type=int, default=batch_size, help="Embedding 批大小")
    p.add_argument("--no_jieba", action="store_true", help="不使用 jieba（强制使用 char ngram）")
    p.add_argument("--log_level", default="INFO", help="日志等级：DEBUG/INFO/WARN/ERROR")
    return p.parse_args()


def main() -> None:
    load_dotenv()  # 读取 .env 参数（若存在）
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s | %(levelname)s | %(message)s")
    run(
        input_path=args.input,
        provider=args.provider,
        model=args.model,
        output_dir=args.output,
        k_min=args.k_min,
        k_max=args.k_max,
        batch_size=args.batch_size,
        use_jieba=(not args.no_jieba),
    )


if __name__ == "__main__":
    main()
