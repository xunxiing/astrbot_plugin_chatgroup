from __future__ import annotations

import os
import asyncio
import shutil
import time
from typing import Any, List

import numpy as np

from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.event.filter import EventMessageType
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

from core.clipper import ContextClipper
from core.decider import SpeakDecider
from core.orchestrator import Orchestrator, AstrBotEmbeddingAdapter
from core.continue_flow import ContinuePolicy
import json


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
        plugin_dir = os.path.dirname(os.path.abspath(__file__))
        self.config = config or {}

        # Apply config overrides to environment for internal components
        self._apply_config_env(self.config)

        # Resolve persistent data dir under AstrBot/data
        store_dir = _resolve_data_store_dir(plugin_dir)
        # Migration is intentionally disabled

        per_chat_cap = int(os.getenv("CHATGROUP_PER_CHAT_CAP", "400"))
        llm_cfg = (self.config.get("llm") or {}) if isinstance(self.config, dict) else {}
        llm_provider_id = llm_cfg.get("provider_id") if isinstance(llm_cfg, dict) else None
        llm_model = llm_cfg.get("model") if isinstance(llm_cfg, dict) else None
        emb_cfg = (self.config.get("embedding") or {}) if isinstance(self.config, dict) else {}
        emb_provider_id = emb_cfg.get("provider_id") if isinstance(emb_cfg, dict) else None
        emb_model = emb_cfg.get("model") if isinstance(emb_cfg, dict) else None
        embedder = AstrBotEmbeddingAdapter(self.context, prefer_provider_id=emb_provider_id, prefer_model=emb_model)
        self.orchestrator = Orchestrator(store_dir, per_chat_cap=per_chat_cap, embedder=embedder)
        self.clipper = ContextClipper(
            max_chars=int(os.getenv("CHATGROUP_TRIM_MAX_CHARS", "6000")),
            max_messages=int(os.getenv("CHATGROUP_TRIM_MAX_MSGS", "30")),
        )
        self.decider = SpeakDecider()

        # Continue-discussion policy (decoupled core)
        self._test_log_path = os.path.join(_resolve_data_store_dir(plugin_dir), "continue_test.log")
        self.continue_policy = ContinuePolicy(self.orchestrator, self.orchestrator.embedder, self._test_log_path)

        # Optionally apply LLM provider selection via AstrBot ProviderManager
        if isinstance(llm_provider_id, str) and llm_provider_id.strip():
            async def _apply_llm_choice():
                try:
                    from astrbot.core.provider.entities import ProviderType  # type: ignore
                    await self.context.provider_manager.set_provider(llm_provider_id.strip(), ProviderType.CHAT_COMPLETION)
                    prov = self.context.get_using_provider()
                    if prov and isinstance(llm_model, str) and llm_model.strip():
                        try:
                            prov.set_model(llm_model.strip())
                        except Exception:
                            pass
                except Exception:
                    pass

            # Register async task for provider selection
            try:
                self.context.register_task(_apply_llm_choice(), "chatgroup:apply llm selection")
            except Exception:
                # Fallback: fire-and-forget
                try:
                    asyncio.create_task(_apply_llm_choice())
                except Exception:
                    pass

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

    # ---- Hooks ----
    @filter.event_message_type(EventMessageType.ALL)
    async def on_message_collect(self, event: AstrMessageEvent):
        """Collect recent messages into vector store for topic grouping.

        Stores plain text user messages. Skips empty and the topic command itself.
        """
        # Determine chat id
        chat_id = None
        try:
            chat_id = event.get_group_id() or None
        except Exception:
            chat_id = None
        if not chat_id:
            chat_id = getattr(event, "unified_msg_origin", None) or "default"

        text = getattr(event, "message_str", None)
        if not isinstance(text, str) or not text.strip():
            return
        t = text.strip()
        # Skip our own command trigger words to reduce noise
        if t == "讨论" or t == "#讨论":
            return

        try:
            sender = None
            try:
                sender = event.get_sender_name()
            except Exception:
                sender = None
            await self.orchestrator.record_message(chat_id=chat_id, text=t, sender=sender, ts=int(time.time()))
        except Exception:
            pass

    @filter.on_llm_request()
    async def on_llm_request(self, event: AstrMessageEvent, req: Any):
        """On LLM request: store embedding for latest user message and trim contexts."""
        # Extract chat id (group or private session)
        chat_id = None
        try:
            chat_id = event.get_group_id() or None
        except Exception:
            chat_id = None
        if not chat_id:
            chat_id = getattr(event, "unified_msg_origin", None) or "default"

        # Try to locate the latest user message text
        latest_text = getattr(event, "message_str", None)
        contexts = getattr(req, "contexts", None)
        if contexts and isinstance(contexts, list):
            # Prefer the last 'user' role in contexts
            for m in reversed(contexts):
                if isinstance(m, dict) and m.get("role") == "user":
                    latest_text = m.get("content") or latest_text
                    break

        # Persist embedding for the latest user text
        if latest_text and isinstance(latest_text, str) and self.decider.should_process(latest_text):
            try:
                sender = None
                try:
                    sender = event.get_sender_name()
                except Exception:
                    sender = None
                await self.orchestrator.record_message(chat_id=chat_id, text=latest_text, sender=sender, ts=int(time.time()))
            except Exception:
                pass

        # Mark pending for continue-discussion
        if chat_id:
            self.continue_policy.mark_pending(chat_id)

        # Trim contexts to keep prompt compact
        if contexts and isinstance(contexts, list):
            try:
                req.contexts = self.clipper.trim(contexts)
            except Exception:
                pass

    # ---- Continue-discussion (降级：撤回/缩短/补充) ----
    @filter.on_decorating_result()
    async def on_decorating_result_continue(self, event: AstrMessageEvent):
        """Before sending message: degrade LLM replies in busy threads.

        Logic per 分析.md：基于近邻消息的时间与相似度，做 撤回/缩短/补充/继续 决策。
        仅在检测到该会话刚经历了 on_llm_request 时触发，以尽量只作用于 LLM 回复。
        """
        # Identify chat id (prefer group)
        chat_id = None
        try:
            chat_id = event.get_group_id() or None
        except Exception:
            chat_id = None
        if not chat_id:
            chat_id = getattr(event, "unified_msg_origin", None) or "default"

        # Only act if there was a recent LLM request in this chat
        if not self.continue_policy.should_handle(chat_id):
            return

        # Extract candidate text from the outgoing result chain
        try:
            result = event.get_result()
        except Exception:
            result = None
        if not result:
            return
        chain = getattr(result, "chain", None)
        if not isinstance(chain, list) or not chain:
            return

        candidate_text = self._chain_plain_text(chain)
        if not candidate_text or not candidate_text.strip():
            # Nothing to analyze
            self.continue_policy.clear_pending(chat_id)
            return

        # Decide action
        try:
            decision, debug = await self.continue_policy.decide(chat_id, candidate_text)
        except Exception:
            # Fail-open
            self.continue_policy.clear_pending(chat_id)
            return

        # Apply decision
        try:
            if decision == "withdraw":
                # 撤回：不发。尽量在装饰阶段直接取消发送。
                try:
                    event.stop_event()
                except Exception:
                    # Fallback: replace with a lightweight acknowledgement
                    self._set_chain_text(result, "👍")
            elif decision == "shorten":
                short = self.continue_policy.shorten_text(candidate_text)
                self._set_chain_text(result, short)
            elif decision == "supplement":
                supp = self.continue_policy.supplement_text(candidate_text)
                self._set_chain_text(result, supp)
            else:
                # continue: no change
                pass
        finally:
            # Clear pending flag regardless of action
            self.continue_policy.clear_pending(chat_id)

    # ---- Helpers ----
    def _chain_plain_text(self, chain: List[Any]) -> str:
        parts: List[str] = []
        for seg in chain:
            try:
                if isinstance(seg, Plain):
                    parts.append(str(getattr(seg, "text", "") or getattr(seg, "content", "")))
                else:
                    # Best effort: some components may have text
                    t = getattr(seg, "text", None) or getattr(seg, "content", None)
                    if isinstance(t, str):
                        parts.append(t)
            except Exception:
                continue
        return "\n".join([p for p in parts if p]).strip()

    def _set_chain_text(self, result_obj: Any, text: str) -> None:
        # Replace the entire chain with a single Plain
        try:
            result_obj.chain = [Plain(text)]
        except Exception:
            # Fallback: try to mutate existing
            try:
                chain = getattr(result_obj, "chain", None)
                if isinstance(chain, list) and chain:
                    if isinstance(chain[0], Plain):
                        chain[0].text = text
                        result_obj.chain = [chain[0]]
                    else:
                        result_obj.chain = [Plain(text)]
            except Exception:
                pass

    # moved core decision helpers into core/continue_flow.py to decouple

    # ---- Commands ----
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

        topics = self.orchestrator.list_topics(chat_id)
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
