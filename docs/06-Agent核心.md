# 第 6 章 Agent 核心：一个循环，七样配置

> 本章代码：`chapters/ch06_agent/01_agent_anatomy.py`、`02_persistence.py`、`03_config_limits.py`

## 6.1 本章目标

- 说清 `create_agent` 的每个参数在做什么
- 用 `checkpointer` + `thread_id` 实现多轮对话与会话隔离
- 用 `recursion_limit` 给 Agent 装上"刹车"

## 6.2 Agent 的公式

```
Agent = 模型 + 工具 + 系统提示词 + 状态 + 上下文 + 中间件 + 结构化输出
```

`create_agent` 的参数就是这七样：

| 参数 | 作用 | 默认 |
| --- | --- | --- |
| `model` | 字符串 `"provider:model"` 或模型实例 | 必填 |
| `tools` | 函数、`BaseTool`、工具字典的列表 | `[]` |
| `system_prompt` | 字符串或 `SystemMessage` | 无 |
| `state_schema` | 扩展状态（如加计数器） | `AgentState` |
| `context_schema` | 每次调用传入的上下文结构 | 无 |
| `middleware` | 中间件列表（第 7 章） | `[]` |
| `response_format` | 结构化输出（第 4 章） | 无 |
| `checkpointer` | 持久化后端（本章） | 无 |
| `store` | 长期记忆（第 8 章） | 无 |
| `name` | Agent 标识，作为子图时有用 | 无 |

## 6.3 输入输出协议

```python
result = agent.invoke({"messages": [{"role": "user", "content": "..."}]})
```

输入是 **state 的增量**，`messages` 会被 `add_messages` reducer 追加而不是覆盖。

输出是完整 state，常用的三个键：

| 键 | 内容 |
| --- | --- |
| `messages` | 完整对话（含工具调用与返回） |
| `structured_response` | 配了 `response_format` 才有 |
| `__interrupt__` | 触发了人工审批/中断才有（第 7 章） |

`01_agent_anatomy.py` 里打印了一次完整轨迹：

```
0. HumanMessage  帮我看看 11 月 5 日北京到成都的航班...
1. AIMessage     调工具 search_flights(...), get_weather(...)   <- 并行调用
2. ToolMessage   北京->成都 2025-11-05: ...
3. ToolMessage   成都：多云，18°C
4. AIMessage     调工具 TripPlan(...)                            <- 结构化收口
5. ToolMessage   Returning structured response: ...
```

注意第 1 步：**模型一次返回两个 tool_calls，框架会自动并发执行**。这是 Agent 比手写串行循环快的主要原因。

## 6.4 多轮对话：checkpointer + thread_id

模型无状态，Agent 默认也无状态。要"记住上一轮"，两件事缺一不可：

```python
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(..., checkpointer=InMemorySaver())
cfg = {"configurable": {"thread_id": "session-A"}}

agent.invoke({"messages": [HumanMessage("我叫林墨")]}, config=cfg)
agent.invoke({"messages": [HumanMessage("我叫什么？")]}, config=cfg)  # 记得
```

| 概念 | 含义 | 类比 |
| --- | --- | --- |
| `thread_id` | 一条会话 | 微信里的一个聊天窗口 |
| `checkpointer` | 存会话的地方 | 聊天记录数据库 |
| `get_state(cfg)` | 读回当前状态 | 打开聊天记录 |

换 `thread_id` 就是另一段独立会话，历史完全隔离。这是多用户系统的隔离机制——**千万别共用一个 thread_id**。

`get_state()` 返回的 `StateSnapshot` 里，`values` 是状态数据，`next` 是下一个待执行节点（空元组表示已结束）。

## 6.5 生产环境把记忆放哪

| 后端 | 包 | 适用 |
| --- | --- | --- |
| `InMemorySaver` | 内置 | 开发、单测 |
| `SqliteSaver` | `langgraph-checkpoint-sqlite` | 单机服务、本地工具 |
| `PostgresSaver` | `langgraph-checkpoint-postgres` | 多实例生产环境 |
| `RedisSaver` | `langgraph-checkpoint-redis` | 高并发、短会话 |

**多副本部署必须用外部 checkpointer**。内存版在 3 个副本下会串会话，这个 bug 排查起来非常痛苦。

## 6.6 给 Agent 装刹车：recursion_limit

Agent 最危险的不是笨，是停不下来：工具一直失败、模型反复重试，账单就上去了。

```python
agent.invoke({...}, config={"recursion_limit": 12})
```

超了会抛 `GraphRecursionError`。默认 25 个超步，生产建议 10~30。

```python
try:
    out = agent.invoke({...}, config={"recursion_limit": 12})
except GraphRecursionError:
    return "这个问题比较复杂，我先整理一下已有信息，稍后给你完整答复。"
```

**必须捕获**，否则用户看到的是 500。

更细的限流手段在第 7 章：`ToolCallLimitMiddleware`（限工具次数）、`ModelCallLimitMiddleware`（限模型调用次数）。

## 6.7 config 里还能放什么

| 键 | 用途 |
| --- | --- |
| `configurable.thread_id` | 会话标识 |
| `recursion_limit` | 最大超步 |
| `tags` | 追踪系统里筛选（`["prod", "refund"]`） |
| `metadata` | 业务标签（`{"user_id": "u_1"}`） |
| `run_name` | LangSmith 里显示的名字 |
| `callbacks` | 回调处理器（第 12 章） |

`context` 是另一回事：它在 `invoke(context=...)` 里传，进的是工具和中间件，不进模型上下文。分清 `thread_id`（会话）和 `context`（本次调用的业务数据）。

## 6.8 常见坑

| 现象 | 原因 | 解决 |
| --- | --- | --- |
| 记不住上文 | 没挂 checkpointer 或 thread_id 变了 | 两者都要 |
| 用户 A 看到用户 B 的历史 | 共用 thread_id | 每用户一个 |
| 进程重启记忆消失 | 用了 InMemorySaver | 换 SQLite/Postgres |
| 多副本串会话 | 内存 checkpointer | 外部存储 |
| 偶发 `GraphRecursionError` | 工具反复失败 | 捕获 + 限流中间件 |

## 6.9 练习

1. 用 `SqliteSaver` 改写 `02_persistence.py`，杀掉进程后重新运行，验证历史还在。
2. 给 Agent 加 `state_schema`，统计"本次会话调了几次工具"，并在最后打印。
3. 把 `recursion_limit` 依次设为 2/3/5，找出这个任务的最小可用值。
4. 同时用两个 `thread_id` 交替提问，验证历史互不干扰。

下一章：中间件。这是 v1 的灵魂，也是把"能跑"变成"能上线"的关键。
