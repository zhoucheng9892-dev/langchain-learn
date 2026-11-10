# 生产部署与成本控制

## 步数与成本上限

Agent 默认的 `recursion_limit` 是 25 个超步。生产环境应当显式设置更小的值，常见区间是 10 到 30，取决于任务复杂度。超过上限会抛 `GraphRecursionError`，务必捕获并给出兜底话术，而不是直接返回 500。

成本主要来自三处：模型 token、工具调用的外部 API、以及失败重试。控制手段包括：给 `max_tokens` 设上限、用 `ToolCallLimitMiddleware` 限制工具次数、用 `ModelCallLimitMiddleware` 限制模型调用次数、开启供应商侧的前缀缓存。

## 延迟优化

- 能用异步就用异步，`ainvoke` 与 `asyncio.gather` 可以把多路请求的等待时间重叠。
- 工具调用尽量并行，模型一次返回多个 tool_calls 时框架会自动并发执行。
- 流式输出首字延迟，`stream_mode="messages"` 让用户在第一秒就看到反馈。

## 可观测性

三种手段配合使用：本地用 `BaseCallbackHandler` 统计 token 与耗时；线上接 LangSmith 做全链路追踪；关键节点用结构化日志记录 thread_id、user_id、工具名与耗时。评测方面，准备 20 到 50 条覆盖典型场景的数据集，用 LLM-as-judge 打分，每次改提示词或换模型都跑一遍回归。

## 部署形态

最简单的是把 Agent 包成 FastAPI 服务，`/chat` 返回 SSE 流。需要持久化会话时挂 PostgresSaver；需要多实例部署时必须用外部 checkpointer，内存版在多副本下会串会话。LangGraph Platform 提供托管方案，自带 checkpointer、任务队列与可观测性，适合不想自己维护基础设施的团队。
