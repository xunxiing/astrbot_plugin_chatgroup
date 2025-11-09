from __future__ import annotations

import os
import re
import asyncio
import shutil
import time
from typing import Any, List, Dict, Tuple

import numpy as np

from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.event.filter import EventMessageType, event_message_type
from astrbot.api.star import Context, Star, register
from astrbot.api.message_components import Plain
try:
    from astrbot.api import AstrBotConfig  # type: ignore
except Exception:  # pragma: no cover
    AstrBotConfig = dict  # fallback for type hints

# Ensure plugin directory is on sys.path so local packages (core) can be imported
import sys
_PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
if _PLUGIN_DIR not in sys.path:
    sys.path.insert(0, _PLUGIN_DIR)

try:
    from core.clipper import ContextClipper  # type: ignore
    from core.decider import SpeakDecider  # type: ignore
    from core.orchestrator import Orchestrator, AstrBotEmbeddingAdapter  # type: ignore
    from core.continue_flow import ContinuePolicy  # type: ignore
except Exception:
    ContextClipper = None  # type: ignore
    SpeakDecider = None  # type: ignore
    Orchestrator = None  # type: ignore
    AstrBotEmbeddingAdapter = None  # type: ignore
    ContinuePolicy = None  # type: ignore
import json
import tempfile
from datetime import datetime
import sqlite3
import hashlib
from pathlib import Path


def _ensure_json_serializable(obj):
    """递归确保对象是 JSON 可序列化的，转换 NumPy 数组和其他不可序列化的对象"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: _ensure_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_ensure_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(_ensure_json_serializable(item) for item in obj)
    else:
        return obj


def _resolve_data_store_dir(plugin_dir: str) -> str:
    """Resolve a persistent store directory under AstrBot/data, not the plugin folder.

    Priority:
    1) CHATGROUP_STORE_DIR (explicit override)
    2) ASTRBOT_DATA_DIR (AstrBot data root) + /chatgroup
    3) Two-level up from plugin_dir (…/AstrBot/data) + /chatgroup
    """
    override = os.getenv("CHATGROUP_STORE_DIR")
    if override:
        return os.path.abspath(override)

    data_root = os.getenv("ASTRBOT_DATA_DIR")
    if not data_root:
        data_root = os.path.abspath(os.path.join(plugin_dir, os.pardir, os.pardir))
    return os.path.join(data_root, "chatgroup")


def _migrate_legacy_store(plugin_dir: str, new_store_dir: str) -> None:
    """No-op: migration disabled per user request."""
    return


@register(
    "chatgroup",
    "codex",
    "讨论组工作流：上下文裁剪 + 语义向量存储 + 话题聚类",
    "0.1.0",
    ""
)
class ChatGroupPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig | None = None):
        super().__init__(context)
        self.config = config or {}
        try:
            self._apply_config_env(self.config)
        except Exception:
            pass

        # recording configs
        self._record_enabled = str(os.getenv("CHATGROUP_RECORD", "1")).strip() not in {"0", "false", "False"}
        try:
            self._retention_sec = int(os.getenv("CHATGROUP_RETENTION_SEC", "604800"))  # 7 days
        except Exception:
            self._retention_sec = 7 * 24 * 3600

        # Initialize embedding cache store (by text hash), with robust fallback
        store_root = _resolve_data_store_dir(_PLUGIN_DIR)
        self._text_embed_store = self._init_embedding_store(store_root)

    @staticmethod
    def _apply_config_env(cfg: dict) -> None:
        # Limits
        limits = cfg.get("limits") or {}
        if isinstance(limits.get("per_chat_cap"), int):
            os.environ["CHATGROUP_PER_CHAT_CAP"] = str(limits["per_chat_cap"])
        if isinstance(limits.get("trim_max_chars"), int):
            os.environ["CHATGROUP_TRIM_MAX_CHARS"] = str(limits["trim_max_chars"])
        if isinstance(limits.get("trim_max_msgs"), int):
            os.environ["CHATGROUP_TRIM_MAX_MSGS"] = str(limits["trim_max_msgs"])
        # LLM provider preferences are not applied via env; AstrBot manages providers centrally.
    
    def _init_embedding_store(self, store_root: str):
        """Create the text embedding store with fallbacks to writable paths.

        Priority:
        1) <ASTRBOT_DATA_DIR or plugin/../../>/chatgroup/vec_cache
        2) %TEMP%/astrbot_chatgroup/vec_cache
        Returns a _TextEmbeddingStore instance or None if all attempts fail.
        """
        # Primary path
        try:
            primary_dir = os.path.join(store_root, "vec_cache")
            os.makedirs(primary_dir, exist_ok=True)
            store = _TextEmbeddingStore(os.path.join(primary_dir, "text_embeds.sqlite"))
            # touch DB to ensure it is openable
            _ = store.has("__ping__")
            self._vec_cache_dir = primary_dir
            return store
        except Exception as e:
            print(f"chatgroup: primary vec store init failed: {e}")
        # Fallback to temp
        try:
            import tempfile as _tf
            fallback_dir = os.path.join(_tf.gettempdir(), "astrbot_chatgroup", "vec_cache")
            os.makedirs(fallback_dir, exist_ok=True)
            store = _TextEmbeddingStore(os.path.join(fallback_dir, "text_embeds.sqlite"))
            _ = store.has("__ping__")
            self._vec_cache_dir = fallback_dir
            return store
        except Exception as e:
            print(f"chatgroup: fallback vec store init failed: {e}")
            return None

    # ---- Commands ----
    @event_message_type(EventMessageType.ALL)
    async def _record_inbox_message(self, event: AstrMessageEvent):
        """把平台原始消息入库到 PlatformMessageHistory，供本插件聚类使用。

        - 每条原始消息（文本/表情/引用等）按消息段列表存为 JSON（与 OneBot 结构兼容）。
        - 会话键采用 (platform_id, user_id)，其中群聊以 group_id 作 user_id，私聊以 sender_id 作 user_id。
        - 可通过环境变量 `CHATGROUP_RECORD=0` 关闭。
        - 可用 `CHATGROUP_RETENTION_SEC` 控制保留时长（默认 7 天）。
        """
        if not self._record_enabled:
            return

        # 获取平台/会话标识
        try:
            platform_id = event.get_platform_id()
        except Exception:
            platform_id = None
        if not platform_id:
            return

        # 对于群聊，用 group_id 作为 user_id；否则用 sender_id
        try:
            group_id = event.get_group_id() or ""
        except Exception:
            group_id = ""
        try:
            sender_id = event.get_sender_id() or ""
        except Exception:
            sender_id = ""
        user_id = group_id or sender_id
        if not user_id:
            return

        # 取消息段并转换为可序列化结构
        try:
            segs = event.get_messages() or []
        except Exception:
            segs = []
        content_list: list[dict] = []
        for seg in segs:
            try:
                # 组件实现了同步 toDict()；避免重编码大文件，这里用同步版本
                d = seg.toDict() if hasattr(seg, "toDict") else {}
                if d:
                    content_list.append(d)
            except Exception:
                continue

        # 避免空消息写入
        if not content_list:
            # 兼容用纯文本字符串的极端情况
            try:
                msg_str = event.get_message_str() or ""
            except Exception:
                msg_str = ""
            if msg_str.strip():
                content_list = [{"type": "text", "data": {"text": msg_str}}]
            else:
                return

        # 发送者标识（可选）
        try:
            sender_name = event.get_sender_name() or None
        except Exception:
            sender_name = None

        # 入库
        try:
            mhm = self.context.message_history_manager
            await mhm.insert(
                platform_id=platform_id,
                user_id=str(user_id),
                content=content_list,
                sender_id=str(sender_id) if sender_id else None,
                sender_name=sender_name,
            )
            # 清理过旧记录（非阻塞，不影响主流程）
            if self._retention_sec > 0:
                try:
                    await mhm.delete(platform_id=platform_id, user_id=str(user_id), offset_sec=self._retention_sec)
                except Exception:
                    pass

            # Optionally embed each new message once and cache by text content
            try:
                if str(os.getenv("CHATGROUP_EMBED_EVERY_MESSAGE", "1")).strip() not in {"0", "false", "False"}:
                    txt = self._components_to_text(content_list).strip()
                    if txt and self._text_embed_store is not None:
                        await self._ensure_text_embedded(txt)
            except Exception as _e:
                # Do not break message flow on embedding errors
                print(f"chatgroup: embed-on-message failed: {_e}")
        except Exception as e:
            print(f"chatgroup: failed to insert message history: {e}")
            return
    @filter.command("讨论", alias={"#讨论"})
    async def list_topics(self, event: AstrMessageEvent):
        """聚合当前会话最近消息为若干讨论组，并渲染为 HTML 图片返回。"""
        # Identify chat id
        chat_id = None
        try:
            chat_id = event.get_group_id() or None
        except Exception:
            chat_id = None
        if not chat_id:
            chat_id = getattr(event, "unified_msg_origin", None) or "default"

        # 调用 src 聚类逻辑一次
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except Exception:
            pass

        try:
            from src.main import cluster_once
        except Exception as e:
            yield event.plain_result(f"无法加载聚类逻辑: {e}")
            return

        # Prefer reading recent messages from AstrBot DB and dump to a temp jsonl
        input_path = None
        try:
            platform_id = event.get_platform_id()
            # for group chats use group_id, otherwise fallback to sender_id
            user_id = ""
            try:
                user_id = event.get_group_id()
            except Exception:
                user_id = ""
            if not user_id:
                try:
                    user_id = event.get_sender_id()
                except Exception:
                    user_id = ""
            if platform_id and user_id:
                input_path = await self._dump_recent_messages_jsonl(
                    platform_id=platform_id,
                    user_id=user_id,
                    max_rows=int(os.getenv("CHATGROUP_DB_LIMIT", "500")),
                    channel_hint=str(chat_id),
                )
        except Exception as e:
            # DB not available or failed; fallback to configured file
            print(f"chatgroup: failed to export DB messages, fallback to file: {e}")

        if not input_path:
            input_path = os.getenv("INPUT", "./chat.jsonl")
        # If fallback path does not exist, bail out gracefully
        try:
            if not os.path.exists(input_path):
                yield event.plain_result("暂无记录，等待新消息后再试~")
                return
        except Exception:
            pass
        provider = os.getenv("PROVIDER", "siliconflow")
        model = os.getenv("MODEL", "BAAI/bge-m3")
        k_min = int(os.getenv("K_MIN", "2"))
        k_max = int(os.getenv("K_MAX", "12"))
        batch_size = int(os.getenv("BATCH_SIZE", "64"))
        use_jieba = True

        try:
            # Read texts back from temp jsonl
            texts: List[str] = []
            with open(input_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        t = (rec.get("text") or "").strip()
                        if t:
                            texts.append(t)
                    except Exception:
                        continue
            if not texts:
                yield event.plain_result("暂无记录，等待新消息后再试~")
                return

            # Fetch cached embeddings; embed missing via adapter if present
            vecs_map: Dict[str, np.ndarray] = {}
            missing_items: List[Tuple[str, str]] = [(t, _sha256_text(t)) for t in texts]
            if self._text_embed_store is not None:
                vecs_map, missing_items = self._text_embed_store.bulk_get(texts)
            if missing_items:
                prov = self._pick_embedding_provider()
                if prov is None:
                    yield event.plain_result("未检测到可用的嵌入(Embedding)提供商，请在 AstrBot 中配置后重试。")
                    return
                miss_texts = [it[0] for it in missing_items]
                try:
                    new_vecs = await prov.get_embeddings(miss_texts)
                except Exception:
                    new_vecs = []
                    for t in miss_texts:
                        try:
                            v = await prov.get_embedding(t)
                            new_vecs.append(v)
                        except Exception:
                            new_vecs.append(None)
                for (t, sha), v in zip(missing_items, new_vecs):
                    if v is None:
                        continue
                    arr = np.asarray(v, dtype=np.float32)
                    self._text_embed_store.put(sha, arr)
                    vecs_map[sha] = arr

            # Build X if all available and pre-seed cache file for src.cluster_once
            X_list: List[np.ndarray] = []
            all_present = True
            for t in texts:
                sha = _sha256_text(t)
                arr = vecs_map.get(sha)
                if arr is None:
                    all_present = False
                    break
                X_list.append(arr.astype(np.float32))

            if all_present:
                cache_path = _compute_cache_path(provider, model, texts)
                Path(os.path.dirname(cache_path)).mkdir(parents=True, exist_ok=True)
                np.save(cache_path, np.vstack(X_list))
            # No direct provider call here; we rely on the preseeded cache.

            clusters = cluster_once(
                input_path=input_path,
                provider=provider,
                model=model,
                k_min=k_min,
                k_max=k_max,
                batch_size=batch_size,
                use_jieba=use_jieba,
            )
        except Exception as e:
            yield event.plain_result(f"聚类失败: {e}")
            return

        topics = []
        for c in clusters:
            examples = [
                {"sender": (m.get("user") or "用户"), "text": (m.get("text") or "")}
                for m in (c.get("representative_messages") or [])
            ][:4]
            topics.append(
                {
                    "topic": c.get("topic") or f"主题#{c.get('cluster_id')}",
                    "size": int(c.get("size") or 0),
                    "keywords": c.get("keywords") or [],
                    "examples": examples,
                }
            )
        if not topics:
            yield event.plain_result("暂无记录，等待新消息后再试~")
            return

        # Build HTML and render to image via AstrBot renderer
        tmpl = """
<div style=\"width: 1024px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'PingFang SC', 'Noto Sans CJK SC', 'Microsoft YaHei', Arial, sans-serif; background: #fff; color: #222; padding: 24px;\">
  <h1 style=\"margin-top: 0; font-size: 28px;\">当前讨论</h1>
  <div style=\"display: grid; grid-template-columns: 1fr 1fr; gap: 16px;\">
  {% for t in topics %}
    <div style=\"border: 1px solid #eee; border-radius: 8px; padding: 12px;\">
      <div style=\"font-size: 18px; font-weight: 700; margin-bottom: 6px;\">{{ t.topic }}</div>
      <div style=\"font-size: 12px; color: #666; margin-bottom: 8px;\">条目: {{ t.size }} | 关键词: {{ ", ".join(t.keywords[:6]) }}</div>
      <ul style=\"margin: 0; padding-left: 18px;\">
      {% for ex in t.examples %}
        <li style=\"margin: 6px 0; line-height: 1.35;\"> 
          <span style=\"color:#555;\">{{ ex.sender or \"用户\" }}</span>：
          <span>{{ ex.text[:80] }}{% if ex.text|length > 80 %}…{% endif %}</span>
        </li>
      {% endfor %}
      </ul>
    </div>
  {% endfor %}
  </div>
</div>
        """

        try:
            # 确保所有数据都是 JSON 可序列化的
            serializable_topics = _ensure_json_serializable(topics)
            url = await self.html_render(tmpl, {"topics": serializable_topics})
            yield event.image_result(url)
        except Exception as e:
            # 记录错误以便调试
            print(f"HTML渲染失败: {e}")
            # Fallback: plain text rendering
            lines: List[str] = []
            for t in topics:
                lines.append(f"【{t['topic']}】条目 {t['size']} | 关键词 {', '.join(t['keywords'][:6])}")
            yield event.plain_result("\n".join(lines))

    async def _dump_recent_messages_jsonl(
        self,
        platform_id: str,
        user_id: str,
        max_rows: int = 500,
        channel_hint: str | None = None,
    ) -> str | None:
        """Read recent messages from AstrBot DB and convert to a clustering-friendly jsonl file.

        Returns a temp jsonl path, or None if no data/failed.
        """
        try:
            mhm = self.context.message_history_manager
        except Exception:
            return None

        try:
            rows = await mhm.get(
                platform_id=platform_id,
                user_id=user_id,
                page=1,
                page_size=max_rows,
            )
        except Exception:
            return None

        if not rows:
            return None

        tmp_dir = tempfile.gettempdir()
        ts_tag = int(time.time())
        safe_pf = str(platform_id).replace(os.sep, "_").replace(":", "_")
        safe_uid = str(user_id).replace(os.sep, "_").replace(":", "_")
        out_path = os.path.join(tmp_dir, f"chatgroup_{safe_pf}_{safe_uid}_{ts_tag}.jsonl")

        def _as_iso(dt: Any) -> str | None:
            try:
                if isinstance(dt, str):
                    return dt
                if isinstance(dt, datetime):
                    return dt.isoformat()
                return None
            except Exception:
                return None

        def _segments_to_text(segs: list) -> str:
            parts: list[str] = []
            for s in segs:
                try:
                    t = (s.get("type") or "").lower() if isinstance(s, dict) else ""
                    data = s.get("data") if isinstance(s, dict) else None
                    if t in ("text", "plain"):
                        txt = ""
                        if isinstance(data, dict) and isinstance(data.get("text"), str):
                            txt = data.get("text")
                        elif isinstance(s, dict) and isinstance(s.get("text"), str):
                            txt = s.get("text")
                        if txt:
                            parts.append(txt)
                    elif t in ("image", "video", "record", "audio"):
                        parts.append("[媒体]")
                    elif t in ("reply", "forward"):
                        parts.append("[引用]")
                    elif t in ("at", "atall"):
                        parts.append("[@]")
                    elif t in ("face", "emoji", "wechattemoji", "wechatemoji"):
                        parts.append("[表情]")
                except Exception:
                    continue
            return " ".join([p for p in parts if p]).strip()

        def _flatten_content_to_text(content: Any) -> str:
            try:
                if isinstance(content, dict):
                    if isinstance(content.get("message"), str):
                        return str(content.get("message") or "").strip()
                    if isinstance(content.get("text"), str):
                        return str(content.get("text") or "").strip()
                    segs = None
                    if isinstance(content.get("message"), list):
                        segs = content.get("message")
                    elif isinstance(content.get("components"), list):
                        segs = content.get("components")
                    if isinstance(segs, list):
                        return _segments_to_text(segs)
                elif isinstance(content, list):
                    return _segments_to_text(content)
                return str(content)
            except Exception:
                return ""

        try:
            with open(out_path, "w", encoding="utf-8") as f:
                for rec in rows:
                    try:
                        rid = getattr(rec, "id", None)
                        sender_name = getattr(rec, "sender_name", None) or getattr(rec, "user_id", None)
                        created_at = getattr(rec, "created_at", None)
                        content = getattr(rec, "content", None)
                        item = {
                            "id": str(rid) if rid is not None else "",
                            "timestamp": _as_iso(created_at),
                            "user_id": getattr(rec, "sender_id", None) or getattr(rec, "user_id", None),
                            "user_name": sender_name or "用户",
                            "text": _flatten_content_to_text(content),
                            "reply_to": None,
                            "channel": channel_hint or str(user_id),
                            "attachments": None,
                            "html": None,
                        }
                        if not (item["text"] and str(item["text"]).strip()):
                            continue
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
                    except Exception:
                        continue
        except Exception as e:
            print(f"chatgroup: failed to write temp jsonl: {e}")
            return None

        try:
            if os.path.getsize(out_path) == 0:
                return None
        except Exception:
            return None
        return out_path

    # ---- Helpers: embedding adapter + store ----
    def _pick_embedding_provider(self):
        """Select an EmbeddingProvider from AstrBot context.

        Priority:
        1) CHATGROUP_EMBED_PROVIDER_ID matches provider id
        2) First available embedding provider
        Returns the provider object or None if unavailable.
        """
        try:
            all_prov = self.context.get_all_embedding_providers()  # list of EmbeddingProvider
        except Exception:
            return None
        if not all_prov:
            return None
        want = os.getenv("CHATGROUP_EMBED_PROVIDER_ID")
        if want:
            for p in all_prov:
                try:
                    if getattr(p, "provider_id", None) == want or getattr(p, "id", None) == want:
                        return p
                except Exception:
                    continue
        return all_prov[0]

    async def _ensure_text_embedded(self, text: str) -> None:
        sha = _sha256_text(text)
        if self._text_embed_store.has(sha):
            return
        prov = self._pick_embedding_provider()
        if prov is None:
            return
        try:
            vec = await prov.get_embedding(text)
            if vec is not None:
                self._text_embed_store.put(sha, np.asarray(vec, dtype=np.float32))
        except Exception:
            pass

    @staticmethod
    def _components_to_text(components: List[dict]) -> str:
        def _segments_to_text(segs: list) -> str:
            parts: list[str] = []
            for s in segs:
                try:
                    t = (s.get("type") or "").lower() if isinstance(s, dict) else ""
                    data = s.get("data") if isinstance(s, dict) else None
                    if t in ("text", "plain"):
                        txt = ""
                        if isinstance(data, dict) and isinstance(data.get("text"), str):
                            txt = data.get("text")
                        elif isinstance(s, dict) and isinstance(s.get("text"), str):
                            txt = s.get("text")
                        if txt:
                            parts.append(txt)
                    elif t in ("image", "video", "record", "audio"):
                        parts.append("[媒体]")
                    elif t in ("reply", "forward"):
                        parts.append("[引用]")
                    elif t in ("at", "atall"):
                        parts.append("[@]")
                    elif t in ("face", "emoji", "wechattemoji", "wechatemoji"):
                        parts.append("[表情]")
                except Exception:
                    continue
            return " ".join([p for p in parts if p]).strip()
        try:
            return _segments_to_text(components or [])
        except Exception:
            return ""


def _sha256_text(text: str) -> str:
    h = hashlib.sha256()
    h.update(text.encode("utf-8"))
    return h.hexdigest()


def _compute_cache_path(provider: str, model: str, texts: List[str]) -> str:
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

    key = _cache_key(provider, model, texts)
    rel = os.path.join("data", f"embeddings_{_slug(provider)}_{_slug(model)}_{key[:16]}.npy")
    return os.path.abspath(os.path.join(os.getcwd(), rel))


class _TextEmbeddingStore:
    def __init__(self, path: str):
        self.path = path
        Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self):
        return sqlite3.connect(self.path, timeout=30, check_same_thread=False)

    def _init_db(self):
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS text_embeddings (
                    sha TEXT PRIMARY KEY,
                    dim INTEGER NOT NULL,
                    vec BLOB NOT NULL,
                    created_at INTEGER NOT NULL DEFAULT (strftime('%s','now')),
                    updated_at INTEGER NOT NULL DEFAULT (strftime('%s','now'))
                )
                """
            )
            conn.commit()

    def has(self, sha: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute("SELECT 1 FROM text_embeddings WHERE sha=?", (sha,))
            return cur.fetchone() is not None

    def get(self, sha: str) -> np.ndarray | None:
        with self._connect() as conn:
            cur = conn.execute("SELECT vec, dim FROM text_embeddings WHERE sha=?", (sha,))
            row = cur.fetchone()
            if not row:
                return None
            buf, dim = row
            try:
                arr = np.frombuffer(buf, dtype=np.float32)
                if arr.size != int(dim):
                    return None
                return arr
            except Exception:
                return None

    def put(self, sha: str, vec: np.ndarray) -> None:
        v = np.asarray(vec, dtype=np.float32)
        with self._connect() as conn:
            conn.execute(
                "REPLACE INTO text_embeddings(sha, dim, vec, updated_at) VALUES (?,?,?,strftime('%s','now'))",
                (sha, int(v.size), v.tobytes()),
            )
            conn.commit()

    def bulk_get(self, texts: List[str]) -> Tuple[Dict[str, np.ndarray], List[Tuple[str, str]]]:
        """Return (found_map: sha->vec, missing_items: [(text, sha), ...])."""
        shas = [_sha256_text(t) for t in texts]
        found: Dict[str, np.ndarray] = {}
        missing: List[Tuple[str, str]] = []
        if not shas:
            return found, missing
        qmarks = ",".join(["?"] * len(shas))
        try:
            with self._connect() as conn:
                cur = conn.execute(f"SELECT sha, vec, dim FROM text_embeddings WHERE sha IN ({qmarks})", tuple(shas))
                for sha, buf, dim in cur.fetchall():
                    try:
                        arr = np.frombuffer(buf, dtype=np.float32)
                        if arr.size == int(dim):
                            found[sha] = arr
                    except Exception:
                        continue
        except Exception:
            # fallback to individual fetch
            for sha in shas:
                arr = self.get(sha)
                if arr is not None:
                    found[sha] = arr
        for t, sha in zip(texts, shas):
            if sha not in found:
                missing.append((t, sha))
        return found, missing
