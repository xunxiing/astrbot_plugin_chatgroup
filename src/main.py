#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import re
import hashlib
import json
import argparse
import logging
import warnings
from typing import Any, Dict, List

# 抑制常见警告
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="joblib")

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.preprocessing import normalize

from src.data_processing import (
    Message,
    read_messages,
    build_message_text,
    is_low_content,
    has_anchor,
)
from src.embeddings import Embedder, LocalSTEmbedder, SiliconFlowEmbedder
from src.clustering import (
    auto_kmeans,
    hdbscan_cluster,
    bucket_indices_by_time,
    dynamic_min_cluster_size,
    detect_burst_chitchat,
    knn_label_propagation,
    soft_attach_low_content,
)
from src.analysis import ctfidf_terms_for_clusters, make_topic_name, representative_indices
from src.postprocess import (
    attach_continuous_by_typing_speed,
    fill_user_span_within_cluster,
)
from src.utils import ensure_dir, configure_logging, getenv_float, getenv_int


def cluster_once(
    input_path: str,
    provider: str = "siliconflow",
    model: str = "BAAI/bge-m3",
    k_min: int = 2,
    k_max: int = 12,
    batch_size: int = 64,
    use_jieba: bool = True,
) -> List[Dict[str, Any]]:
    """Run clustering once and return clusters.

    Returns a list of clusters with keys:
    cluster_id, topic, size, keywords, representative_messages, message_ids.
    """
    msgs = read_messages(input_path)
    if not msgs:
        return []

    texts = [build_message_text(m) for m in msgs]

    if provider.lower() == "siliconflow":
        embedder: Embedder = SiliconFlowEmbedder(model=model)
    elif provider.lower() == "local":
        embedder = LocalSTEmbedder(model=model)
    else:
        raise RuntimeError("provider 仅支持 siliconflow / local")

    def _slug(s: str) -> str:
        return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("._-")

    def _cache_key(provider: str, model: str, texts: List[str]) -> str:
        h = hashlib.sha256()
        h.update((provider + "\n" + model + "\n").encode("utf-8"))
        sep = "\n\0".encode("utf-8")
        for t in texts:
            if not isinstance(t, str):
                t = str(t)
            h.update(t.encode("utf-8"))
            h.update(sep)
        return h.hexdigest()

    ensure_dir("./data")
    key = _cache_key(provider, model, texts)
    cache_path = os.path.join("data", f"embeddings_{_slug(provider)}_{_slug(model)}_{key[:16]}.npy")

    if os.path.exists(cache_path):
        try:
            X = np.load(cache_path)
            if X.shape[0] != len(texts):
                raise ValueError("cache-size-mismatch")
        except Exception:
            X = embedder.embed(texts, batch_size=batch_size)
            np.save(cache_path, X)
    else:
        X = embedder.embed(texts, batch_size=batch_size)
        np.save(cache_path, X)

    if not np.all(np.isfinite(X)):
        raise RuntimeError("Embedding 中存在 inf/nan，请检查输入或 provider 返回")

    WINDOW_MINUTES = getenv_int("WINDOW_MINUTES", 30)
    H_MIN_CLUSTER = getenv_int("HDBSCAN_MIN_CLUSTER_SIZE", 20)
    H_MIN_SAMPLES = getenv_int("HDBSCAN_MIN_SAMPLES", 5)
    ATTACH_SIM_THR = getenv_float("ATTACH_SIM_THR", 0.35)
    ATTACH_MAX_MINUTES = getenv_int("ATTACH_MAX_MINUTES", 90)
    MAX_RECLUSTER_SIZE = getenv_int("MAX_RECLUSTER_SIZE", 80)

    low_mask = np.array([is_low_content(t) for t in texts], dtype=bool)

    CHITCHAT_SIM_THR = getenv_float("CHITCHAT_SIM_THR", 0.68)
    low_indices = [i for i in range(len(texts)) if low_mask[i]]
    if low_indices:
        chit_center = normalize(np.mean(X[low_indices], axis=0, keepdims=True)).ravel()
    else:
        chit_center = np.zeros(X.shape[1])

    def is_chitchat_vec(vec, center, thr):
        if np.all(center == 0):
            return False
        sim = float((vec @ center).item())
        return sim >= thr

    Xn_all = normalize(X)
    chit_mask = np.array([is_chitchat_vec(Xn_all[i : i + 1], chit_center, CHITCHAT_SIM_THR) for i in range(len(texts))])

    BURST_MIN_USERS = getenv_int("BURST_MIN_USERS", 3)
    BURST_MIN_COUNT = getenv_int("BURST_MIN_COUNT", 3)
    burst_mask = detect_burst_chitchat(
        texts,
        msgs,
        list(range(len(texts))),
        window_minutes=WINDOW_MINUTES,
        min_users=BURST_MIN_USERS,
        min_count=BURST_MIN_COUNT,
    )

    anchor_mask = np.array([has_anchor(t) for t in texts], dtype=bool)
    core_mask = (~low_mask) & ((~(chit_mask | burst_mask)) | anchor_mask)
    border_mask = (~low_mask) & (~core_mask)
    core_mask = core_mask.flatten()
    border_mask = border_mask.flatten()

    valid_idxs = [i for i in range(len(texts)) if core_mask[i]]
    border_idxs = [i for i in range(len(texts)) if border_mask[i]]

    global_labels = np.full(len(texts), -1, dtype=int)
    buckets = bucket_indices_by_time(msgs, valid_idxs, WINDOW_MINUTES)
    next_cid = 0
    for (_ch, _bid), idxs in buckets.items():
        subX = X[idxs]
        mcs = dynamic_min_cluster_size(len(idxs), os.getenv("DYNAMIC_MIN_CLUSTER", "auto"))
        if len(idxs) < max(mcs, 10):
            sub_labels, _ = auto_kmeans(subX, k_min=max(2, k_min), k_max=min(6, len(idxs)))
        else:
            sub_labels, sub_clusterer = hdbscan_cluster(subX, mcs, H_MIN_SAMPLES)
            try:
                probs = sub_clusterer.probabilities_
                thr = getenv_float("HDBSCAN_PROB_THR", 0.38)
                sub_labels = np.where(probs >= thr, sub_labels, -1)
            except Exception:
                pass
            if (sub_labels < 0).mean() > 0.60:
                sub_labels, _ = auto_kmeans(subX, k_min=max(2, k_min), k_max=min(6, len(idxs)))
        unique = sorted(set([l for l in sub_labels if l >= 0]))
        mapping = {l: (next_cid + i) for i, l in enumerate(unique)}
        for j, lab in zip(idxs, sub_labels):
            if lab >= 0:
                global_labels[j] = mapping[lab]
        next_cid += len(unique)

    for c in sorted(set(global_labels)):
        if c < 0:
            continue
        members = np.where(global_labels == c)[0]
        if len(members) > MAX_RECLUSTER_SIZE:
            sub_labels, _ = auto_kmeans(X[members], k_min=2, k_max=min(8, len(members) - 1))
            uniq = sorted(set(sub_labels))
            remap = {l: (next_cid + i) for i, l in enumerate(uniq)}
            for k, lab in zip(members, sub_labels):
                global_labels[k] = remap[lab]
            next_cid += len(uniq)

    for i in range(len(texts)):
        if chit_mask[i] and not anchor_mask[i] and not low_mask[i]:
            global_labels[i] = -2

    cluster_median_ts: Dict[int, int] = {}
    for c in sorted(set(global_labels)):
        if c < 0:
            continue
        cluster_times: List[float] = []
        for i in range(len(global_labels)):
            if global_labels[i] == c and msgs[i].timestamp:
                cluster_times.append(msgs[i].timestamp.timestamp())
        if cluster_times:
            cluster_median_ts[c] = int(np.median(cluster_times))

    cand = [
        i
        for i in range(len(texts))
        if ((global_labels[i] < 0) or border_mask[i])
        and (global_labels[i] != -2)
        and (not low_mask[i])
        and (not (chit_mask[i] and not anchor_mask[i]))
    ]

    KNN_K = getenv_int("KNN_K", 5)
    KNN_MIN_IN_CLUSTER = getenv_int("KNN_MIN_IN_CLUSTER", 2)
    KNN_MEAN_SIM_THR = getenv_float("KNN_MEAN_SIM_THR", 0.34)
    global_labels = knn_label_propagation(
        X,
        global_labels,
        cand,
        msgs,
        k=KNN_K,
        min_in_cluster=KNN_MIN_IN_CLUSTER,
        mean_sim_thr=KNN_MEAN_SIM_THR,
        attach_max_minutes=ATTACH_MAX_MINUTES,
        cluster_median_ts=cluster_median_ts,
    )

    low_idxs = [i for i in range(len(texts)) if low_mask[i]]
    global_labels = soft_attach_low_content(X, global_labels, low_idxs, msgs, attach_sim_thr=ATTACH_SIM_THR)

    ENABLE_CONTINUOUS_CHAIN = int(os.getenv("ENABLE_CONTINUOUS_CHAIN", "1"))
    MAX_TYPING_SPEED_CPS = getenv_float("MAX_TYPING_SPEED_CPS", 10.0)
    labels = attach_continuous_by_typing_speed(
        global_labels,
        msgs,
        texts,
        max_typing_speed_cps=MAX_TYPING_SPEED_CPS,
        enabled=bool(ENABLE_CONTINUOUS_CHAIN),
    )

    ENABLE_USER_SPAN_FILL = int(os.getenv("ENABLE_USER_SPAN_FILL", "1"))
    labels = fill_user_span_within_cluster(labels, msgs, enabled=bool(ENABLE_USER_SPAN_FILL))
    reps = representative_indices(X, labels, topk=5)

    df = pd.DataFrame(
        {
            "id": [m.id for m in msgs],
            "timestamp": [m.timestamp.isoformat() if m.timestamp else None for m in msgs],
            "user_id": [m.user_id for m in msgs],
            "user_name": [m.user_name for m in msgs],
            "channel": [m.channel for m in msgs],
            "text": texts,
            "label": labels,
        }
    )

    clusters: List[Dict[str, Any]] = []
    valid_clusters = [c for c in sorted(df["label"].unique()) if c >= 0]
    docs = [" ".join(df[df.label == c]["text"].tolist()) for c in valid_clusters]
    terms_list = ctfidf_terms_for_clusters(docs, topk=8, use_jieba=use_jieba)
    c2terms = {c: terms for c, terms in zip(valid_clusters, terms_list)}

    for c in valid_clusters:
        sub = df[df["label"] == c]
        terms = c2terms.get(c, [])
        topic = make_topic_name(terms, fallback=f"主题#{c}")
        rep_rows = sub.loc[reps.get(c, [])]
        clusters.append(
            {
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
            }
        )

    return clusters

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
    ensure_dir(output_dir)
    logging.info("读取输入…")
    msgs = read_messages(input_path)
    if not msgs:
        raise SystemExit("没有读到任何消息。请检查 --input")

    logging.info("构建文本…")
    texts = [build_message_text(m) for m in msgs]

    logging.info("向量计算（provider=%s, model=%s）…", provider, model)
    if provider.lower() == "siliconflow":
        embedder: Embedder = SiliconFlowEmbedder(model=model)
    elif provider.lower() == "local":
        embedder = LocalSTEmbedder(model=model)
    else:
        raise SystemExit("provider 仅支持 siliconflow / local")

    # Embedding 缓存：根据 provider+model+texts 生成 key，缓存到 ./data
    def _slug(s: str) -> str:
        return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("._-")

    def _cache_key(provider: str, model: str, texts: List[str]) -> str:
        h = hashlib.sha256()
        h.update((provider + "\n" + model + "\n").encode("utf-8"))
        sep = "\n\0".encode("utf-8")
        for t in texts:
            if not isinstance(t, str):
                t = str(t)
            h.update(t.encode("utf-8"))
            h.update(sep)
        return h.hexdigest()

    ensure_dir("./data")
    key = _cache_key(provider, model, texts)
    cache_path = os.path.join("data", f"embeddings_{_slug(provider)}_{_slug(model)}_{key[:16]}.npy")

    if os.path.exists(cache_path):
        try:
            logging.info("发现缓存，加载向量 %s", cache_path)
            X = np.load(cache_path)
            if X.shape[0] != len(texts):
                logging.warning("缓存条数不匹配，重新计算 embeddings (%d != %d)", X.shape[0], len(texts))
                raise ValueError("cache-size-mismatch")
        except Exception:
            logging.info("缓存不可用，重新计算向量…")
            X = embedder.embed(texts, batch_size=batch_size)
            np.save(cache_path, X)
            logging.info("已写入缓存 %s", cache_path)
    else:
        logging.info("未命中缓存，计算向量…")
        X = embedder.embed(texts, batch_size=batch_size)
        np.save(cache_path, X)
        logging.info("已写入缓存 %s", cache_path)
    if not np.all(np.isfinite(X)):
        raise RuntimeError("Embedding 中存在 inf/nan，请检查输入或 provider 返回值")

    WINDOW_MINUTES = getenv_int("WINDOW_MINUTES", 30)
    H_MIN_CLUSTER = getenv_int("HDBSCAN_MIN_CLUSTER_SIZE", 20)
    H_MIN_SAMPLES = getenv_int("HDBSCAN_MIN_SAMPLES", 5)
    ATTACH_SIM_THR = getenv_float("ATTACH_SIM_THR", 0.35)
    ATTACH_MAX_MINUTES = getenv_int("ATTACH_MAX_MINUTES", 90)
    MAX_RECLUSTER_SIZE = getenv_int("MAX_RECLUSTER_SIZE", 80)

    # 低信息样本（不直接参与聚类）
    low_mask = np.array([is_low_content(t) for t in texts], dtype=bool)

    # 闲聊中心（用低信息均值向量）
    CHITCHAT_SIM_THR = getenv_float("CHITCHAT_SIM_THR", 0.68)
    low_indices = [i for i in range(len(texts)) if low_mask[i]]
    if low_indices:
        chit_center = normalize(np.mean(X[low_indices], axis=0, keepdims=True)).ravel()
    else:
        chit_center = np.zeros(X.shape[1])

    def is_chitchat_vec(vec, center, thr):
        if np.all(center == 0):
            return False
        sim = float((vec @ center).item())
        return sim >= thr

    Xn_all = normalize(X)
    chit_mask = np.array([is_chitchat_vec(Xn_all[i : i + 1], chit_center, CHITCHAT_SIM_THR) for i in range(len(texts))])

    # 同时检测爆发式短语
    BURST_MIN_USERS = getenv_int("BURST_MIN_USERS", 3)
    BURST_MIN_COUNT = getenv_int("BURST_MIN_COUNT", 3)
    burst_mask = detect_burst_chitchat(
        texts,
        msgs,
        list(range(len(texts))),
        window_minutes=WINDOW_MINUTES,
        min_users=BURST_MIN_USERS,
        min_count=BURST_MIN_COUNT,
    )

    anchor_mask = np.array([has_anchor(t) for t in texts], dtype=bool)

    core_mask = (~low_mask) & ((~(chit_mask | burst_mask)) | anchor_mask)
    border_mask = (~low_mask) & (~core_mask)

    core_mask = core_mask.flatten()
    border_mask = border_mask.flatten()

    valid_idxs = [i for i in range(len(texts)) if core_mask[i]]
    border_idxs = [i for i in range(len(texts)) if border_mask[i]]
    logging.info("核心: %d | 边界: %d | 低信息: %d", len(valid_idxs), len(border_idxs), int(low_mask.sum()))

    # 1) 分桶聚类
    logging.info("按 channel + %d 分钟分桶聚类（HDBSCAN 优先）…", WINDOW_MINUTES)
    global_labels = np.full(len(texts), -1, dtype=int)
    buckets = bucket_indices_by_time(msgs, valid_idxs, WINDOW_MINUTES)
    next_cid = 0
    for (_ch, _bid), idxs in buckets.items():
        subX = X[idxs]
        mcs = dynamic_min_cluster_size(len(idxs), os.getenv("DYNAMIC_MIN_CLUSTER", "auto"))
        if len(idxs) < max(mcs, 10):
            sub_labels, _ = auto_kmeans(subX, k_min=max(2, k_min), k_max=min(6, len(idxs)))
        else:
            sub_labels, sub_clusterer = hdbscan_cluster(subX, mcs, H_MIN_SAMPLES)
            try:
                probs = sub_clusterer.probabilities_
                thr = getenv_float("HDBSCAN_PROB_THR", 0.38)
                sub_labels = np.where(probs >= thr, sub_labels, -1)
            except Exception:
                pass
            if (sub_labels < 0).mean() > 0.60:
                sub_labels, _ = auto_kmeans(subX, k_min=max(2, k_min), k_max=min(6, len(idxs)))
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
            sub_labels, _ = auto_kmeans(X[members], k_min=2, k_max=min(8, len(members) - 1))
            uniq = sorted(set(sub_labels))
            remap = {l: (next_cid + i) for i, l in enumerate(uniq)}
            for k, lab in zip(members, sub_labels):
                global_labels[k] = remap[lab]
            next_cid += len(uniq)

    # 进阶：明确标注闲聊，把纯闲聊打成 -2
    for i in range(len(texts)):
        if chit_mask[i] and not anchor_mask[i] and not low_mask[i]:
            global_labels[i] = -2

    # 3.1 kNN 标签传播（对非低信息、非闲聊的未归属/边界样本）
    cluster_median_ts: Dict[int, int] = {}
    for c in sorted(set(global_labels)):
        if c < 0:
            continue
        cluster_times: List[float] = []
        for i in range(len(global_labels)):
            if global_labels[i] == c and msgs[i].timestamp:
                cluster_times.append(msgs[i].timestamp.timestamp())
        if cluster_times:
            cluster_median_ts[c] = int(np.median(cluster_times))

    cand = [
        i
        for i in range(len(texts))
        if ((global_labels[i] < 0) or border_mask[i])
        and (global_labels[i] != -2)
        and (not low_mask[i])
        and (not (chit_mask[i] and not anchor_mask[i]))
    ]

    KNN_K = getenv_int("KNN_K", 5)
    KNN_MIN_IN_CLUSTER = getenv_int("KNN_MIN_IN_CLUSTER", 2)
    KNN_MEAN_SIM_THR = getenv_float("KNN_MEAN_SIM_THR", 0.34)
    global_labels = knn_label_propagation(
        X,
        global_labels,
        cand,
        msgs,
        k=KNN_K,
        min_in_cluster=KNN_MIN_IN_CLUSTER,
        mean_sim_thr=KNN_MEAN_SIM_THR,
        attach_max_minutes=ATTACH_MAX_MINUTES,
        cluster_median_ts=cluster_median_ts,
    )

    # 3.2 低信息样本做质心软吸附兜底
    low_idxs = [i for i in range(len(texts)) if low_mask[i]]
    global_labels = soft_attach_low_content(X, global_labels, low_idxs, msgs, attach_sim_thr=ATTACH_SIM_THR)

    # 3.3 连续发言（按打字速度阈值）补链
    ENABLE_CONTINUOUS_CHAIN = int(os.getenv("ENABLE_CONTINUOUS_CHAIN", "1"))
    MAX_TYPING_SPEED_CPS = getenv_float("MAX_TYPING_SPEED_CPS", 10.0)
    labels = attach_continuous_by_typing_speed(
        global_labels,
        msgs,
        texts,
        max_typing_speed_cps=MAX_TYPING_SPEED_CPS,
        enabled=bool(ENABLE_CONTINUOUS_CHAIN),
    )

    # 3.4 用户时间跨度内补全（每簇每用户：最早~最晚区间）
    ENABLE_USER_SPAN_FILL = int(os.getenv("ENABLE_USER_SPAN_FILL", "1"))
    labels = fill_user_span_within_cluster(labels, msgs, enabled=bool(ENABLE_USER_SPAN_FILL))
    reps = representative_indices(X, labels, topk=5)

    # 主题命名
    logging.info("主题命名（c-TF-IDF）…")
    df = pd.DataFrame(
        {
            "id": [m.id for m in msgs],
            "timestamp": [m.timestamp.isoformat() if m.timestamp else None for m in msgs],
            "user_id": [m.user_id for m in msgs],
            "user_name": [m.user_name for m in msgs],
            "channel": [m.channel for m in msgs],
            "text": texts,
            "label": labels,
        }
    )

    clusters: List[Dict[str, Any]] = []
    valid_clusters = [c for c in sorted(df["label"].unique()) if c >= 0]
    docs = [" ".join(df[df.label == c]["text"].tolist()) for c in valid_clusters]
    terms_list = ctfidf_terms_for_clusters(docs, topk=8, use_jieba=use_jieba)
    c2terms = {c: terms for c, terms in zip(valid_clusters, terms_list)}

    for c in valid_clusters:
        sub = df[df["label"] == c]
        terms = c2terms.get(c, [])
        topic = make_topic_name(terms, fallback=f"主题#{c}")
        rep_rows = sub.loc[reps.get(c, [])]
        clusters.append(
            {
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
            }
        )

    out_jsonl = os.path.join(output_dir, "messages_with_labels.jsonl")
    out_clusters = os.path.join(output_dir, "clusters.json")

    logging.info("写出 %s", out_jsonl)
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")

    logging.info("写出 %s", out_clusters)
    with open(out_clusters, "w", encoding="utf-8") as f:
        json.dump(clusters, f, ensure_ascii=False, indent=2)

    total = len(df)
    noise_cnt = int((df["label"] == -1).sum())
    chatter_cnt = int((df["label"] == -2).sum()) if (-2 in df["label"].unique()) else 0
    border_joined = len([i for i in border_idxs if labels[i] >= 0])
    logging.info(
        "总数: %d | 主题数: %d | 噪声: %d | 闲聊: %d | 边界补回: %d",
        total,
        len([c for c in set(labels) if c >= 0]),
        noise_cnt,
        chatter_cnt,
        border_joined,
    )
    logging.info("完成。共 %d 条消息，被分为 %d 个主题组", len(df), len(clusters))


def parse_args() -> argparse.Namespace:
    input_path = os.getenv("INPUT", "./chat.jsonl")
    provider = os.getenv("PROVIDER", "siliconflow")
    model = os.getenv("MODEL", "BAAI/bge-m3")
    output_dir = os.getenv("OUTPUT", "./out")
    k_min = int(os.getenv("K_MIN", "2"))
    k_max = int(os.getenv("K_MAX", "12"))
    batch_size = int(os.getenv("BATCH_SIZE", "64"))

    p = argparse.ArgumentParser(
        description="一键运行聊天主题分组全流程",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", default=input_path, help="输入 .jsonl 文件或包含 .jsonl 的目录")
    p.add_argument("--provider", default=provider, choices=["siliconflow", "local"], help="向量提供者")
    p.add_argument("--model", default=model, help="向量模型名（siliconflow/local）")
    p.add_argument("--output", default=output_dir, help="输出目录")
    p.add_argument("--k_min", type=int, default=k_min, help="KMeans 最小簇")
    p.add_argument("--k_max", type=int, default=k_max, help="KMeans 最大簇")
    p.add_argument("--batch_size", type=int, default=batch_size, help="Embedding 批大小")
    p.add_argument("--no_jieba", action="store_true", help="不使用 jieba（强制 char ngram）")
    p.add_argument("--log_level", default="INFO", help="日志等级：DEBUG/INFO/WARN/ERROR")
    return p.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    configure_logging(args.log_level)
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
