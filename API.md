# astrbot_plugin_chatgroup - API 文档

本插件提供“讨论组（聊天主题）”相关能力，并对其他插件暴露以下接口：

- 内部 API（跨插件直接调用）
- Web API（通过 AstrBot Dashboard 插件路由 `/api/plug/*` 调用）

说明：所有“会话”均由 `(platform_id, user_id)` 共同标识。

- 群聊：`user_id` 使用群号（或平台群 ID）
- 私聊：`user_id` 使用发送者 ID

## 内部 API（跨插件调用）

其他插件可通过 `Context.get_registered_star()` 获取本插件实例并调用。示例：

```python
# 其它插件内部示例
md = self.context.get_registered_star("chatgroup")
if md and md.star_cls and md.activated:
    chatgroup = md.star_cls  # ChatGroupPlugin 实例
    # 1) 获取当前讨论组
    clusters = await chatgroup.api_list_discussion_groups(
        platform_id="napcat",  # 示例平台 ID
        user_id="123456789",      # 群号或私聊用户 ID
        max_rows=500,              # 可选，默认 500，读取最近 N 条消息
    )
    # 2) 获取指定讨论组详情
    detail = await chatgroup.api_get_discussion_group(
        platform_id="napcat",
        user_id="123456789",
        cluster_id=0,              # cluster_id 来自 clusters 返回值
        max_rows=500,
    )
    # 3) 按时间范围获取讨论组
    clusters2 = await chatgroup.api_list_discussion_groups_by_time(
        platform_id="napcat",
        user_id="123456789",
        start_time="2025-01-01T00:00:00",  # ISO 字符串或时间戳秒数
        end_time=1735737600,                # 也可用时间戳（示例）
        max_rows=1000,
    )
```

### 1) api_list_discussion_groups

- 入参
  - `platform_id: str` 平台 ID（如 `aiocqhttp`、`slack` 等）
  - `user_id: str` 群号或用户 ID
  - `max_rows: int = 500`（可选）读取最近多少条消息后进行聚类
- 出参（List[Dict]）每个元素为一个讨论组：
  - `cluster_id: int` 聚类编号
  - `topic: str` 主题名称
  - `size: int` 组内消息数量
  - `keywords: List[str]` 关键词（最多 8 个）
  - `representative_messages: List[ {id, user, timestamp, text} ]` 代表消息
  - `message_ids: List[str]` 该组包含的所有消息 ID（数据库记录 ID）

### 2) api_get_discussion_group

- 入参
  - `platform_id: str`
  - `user_id: str`
  - `cluster_id: int` 目标讨论组 ID
  - `max_rows: int = 500`（可选）
- 出参（Dict | None）
  - 与 `api_list_discussion_groups` 单个元素结构相同，额外包含：
    - `messages: List[ {id, user_id, user_name, timestamp, text, ...} ]` 组内所有消息详情（从同一批 jsonl 中回表构建）

### 3) api_list_discussion_groups_by_time

- 入参
  - `platform_id: str`
  - `user_id: str`
  - `start_time: str | int | datetime | None` 开始时间（ISO 字符串或时间戳秒数）
  - `end_time: str | int | datetime | None` 结束时间（同上）
  - `max_rows: int = 1000`（可选）读取最近 N 条后在时间范围内过滤
- 出参（List[Dict]）结构同 `api_list_discussion_groups`

备注：内部 API 会尝试利用 AstrBot 的嵌入提供商预先写入缓存，以加速聚类。

## Web API（可选）

所有路由挂载于插件路由前缀 `/api/plug` 下：

- 列出讨论组：
  
  - `GET /api/plug/chatgroup/groups?platform_id=...&user_id=...&max_rows=500`
  - 返回：
    ```json
    {
      "status": "ok",
      "data": { "clusters": [ /* 同内部 API 列表结构 */ ] }
    }
    ```
- 讨论组详情：
  
  - `GET /api/plug/chatgroup/group_detail?platform_id=...&user_id=...&cluster_id=...&max_rows=500`
  - 返回：
    ```json
    {
      "status": "ok",
      "data": { "cluster": { /* 同内部 API 详情结构 */ } }
    }
    ```
- 按时间范围列出讨论组：
  
  - `GET /api/plug/chatgroup/groups_by_time?platform_id=...&user_id=...&start=...&end=...&max_rows=1000`
  - `start`/`end` 支持 ISO 字符串或时间戳秒数
  - 返回：同“列出讨论组”

## 行为与限制

- 讨论组（主题）是基于“最近消息 + 活跃窗口”的无状态计算结果，并非永久存储。
- `max_rows` 控制读取的最近消息上限，建议 200~1000 之间。
- 首次计算如出现嵌入缺失，会尝试调用当前 AstrBot 配置的嵌入提供商进行补齐并缓存。
- 若未配置嵌入提供商，则仍可工作，但可能需要插件 `src/` 的 provider 路径可用或预先有缓存。

## 版本

- 插件版本：`0.1.1+api`（文档声明，不影响 `metadata.yaml`）

