# LangChain 1.x 从入门到精通（DeepSeek 实战版）

一套**可运行**的 LangChain 1.x 中文学习材料：12 章讲义 + 33 个示例程序，全部在 `langchain 1.4.7` + `deepseek-flash` 上实测通过。

不抄官方文档的目录，而是按"学完能干什么"组织：每一章都有能跑的代码、能对照的输出、和真实踩过的坑。

---

## 快速开始

```bash
# 1. 装依赖（需要 uv：https://docs.astral.sh/uv/）
uv sync

# 2. 配密钥
cp .env.example .env      # 填入 DEEPSEEK_API_KEY

# 3. 跑第一个示例
uv run python chapters/ch01_quickstart/01_hello_model.py
```

`.env` 只有两项必填：

```ini
DEEPSEEK_API_KEY=sk-xxx
MODEL_NAME=deepseek-v4.1-flash
```

> **关于模型名**：官方 API 只接受 `deepseek-flash` / `deepseek-v4-pro`。写 `deepseek-v4.1-flash` 会 400。项目里已做别名映射（`src/langchain_learn/llm.py`），所以你写产品名也能跑。

---

## 章节目录

| 章 | 主题 | 关键 API | 示例 |
| --- | --- | --- | --- |
| [00](docs/00-学习路线.md) | 学习路线与环境搭建 | — | — |
| [01](docs/01-快速开始.md) | 快速开始 | `ChatDeepSeek`、`create_agent` | `ch01_quickstart/` (2) |
| [02](docs/02-模型层.md) | 模型层：参数、思考模式、并发 | `init_chat_model`、`reasoning_effort`、`ainvoke` | `ch02_models/` (3) |
| [03](docs/03-消息与Prompt.md) | 消息体系与 Prompt 模板 | `SystemMessage`、`content_blocks`、`ChatPromptTemplate`、`trim_messages` | `ch03_messages/` (3) |
| [04](docs/04-结构化输出.md) | 结构化输出 | `with_structured_output`、`ToolStrategy` | `ch04_structured/` (3) |
| [05](docs/05-工具调用.md) | 工具调用 | `@tool`、`args_schema`、`ToolRuntime` | `ch05_tools/` (3) |
| [06](docs/06-Agent核心.md) | Agent 核心：循环/状态/持久化 | `create_agent`、`InMemorySaver`、`recursion_limit` | `ch06_agent/` (3) |
| [07](docs/07-中间件.md) | 中间件（v1 灵魂） | `before_model`、`wrap_tool_call`、`HumanInTheLoop` | `ch07_middleware/` (3) |
| [08](docs/08-记忆.md) | 记忆：短期/长期/压缩 | `SqliteSaver`、`InMemoryStore`、`SummarizationMiddleware` | `ch08_memory/` (3) |
| [09](docs/09-流式.md) | 流式输出与 SSE | `stream_mode`、`stream_events` | `ch09_streaming/` (2) |
| [10](docs/10-RAG.md) | RAG 检索增强 | `TextSplitter`、`InMemoryVectorStore`、MMR | `ch10_rag/` (3) |
| [11](docs/11-多Agent.md) | 多 Agent 协作 | Agent-as-Tool、`StateGraph` | `ch11_multiagent/` (2) |
| [12](docs/12-工程化.md) | 可观测/评测/成本/迁移 | `BaseCallbackHandler`、`set_llm_cache` | `ch12_production/` (3) |

**三条学习路径**（详见第 0 章）：

- 系统学习：1 → 12，约 12~16 小时
- 应急上线：1 → 5 → 6 → 7 → 12，约 4 小时
- 从 0.x 迁移：1 → 12.7 → 6 → 7，约 2 小时

---

## 这份材料的三个特点

**① 每章都有能跑的代码，不是伪代码**

33 个脚本全部实测，输出与讲义中引用的一致。跑一遍就知道每个 API 的真实行为。

**② 记录了 DeepSeek 的三个专属坑**

| 坑 | 报错 | 解法 | 复现位置 |
| --- | --- | --- | --- |
| 产品名 ≠ 模型名 | `The supported API model names are...` | 用 `deepseek-flash` | `ch02/01_model_init.py` |
| 思考模式禁强制工具 | `Thinking mode does not support this tool_choice` | `reasoning_effort="none"` | `ch04/01_with_structured_output.py` |
| 结构化输出策略 | `This response_format type is unavailable now` | `ToolStrategy(Schema)` | `ch04/02_agent_structured.py` |

**③ 明确区分"v1 写法"和"0.x 残留"**

网上 90% 的教程还是 0.x 的写法（`LLMChain`、`AgentExecutor`、`langchain.chains`）。第 12.7 节给了完整的迁移对照表，第 1 章解释了为什么那些代码在 v1 里会 ImportError。

---

## 项目结构

```
langchain-learn/
├── docs/                    # 13 篇讲义（00 + 12 章）
├── chapters/                # 33 个可运行示例
│   ├── ch01_quickstart/
│   ├── ...
│   └── ch12_production/
├── data/                    # RAG 用的 3 篇示例文档
├── src/langchain_learn/     # 公共模块
│   ├── llm.py               # get_model()：模型工厂 + 名称映射
│   ├── console.py           # 终端输出美化
│   ├── embeddings.py        # 离线确定性 embedding（教学用）
│   └── rag.py               # RAG 构建函数
├── .env.example
├── pyproject.toml
└── uv.lock
```

**为什么公共代码在 `src/` 而不是 `chapters/`**：项目是可编辑安装包，示例里可以直接 `from langchain_learn import get_model`，不需要 `sys.path` 拼路径。这也是真实项目的推荐组织方式。

---

## 全部示例一览

```bash
# 按章运行
uv run python chapters/ch01_quickstart/01_hello_model.py

# 查看某一章所有脚本
ls chapters/ch07_middleware/
```

| 章 | 脚本 | 演示什么 |
| --- | --- | --- |
| 01 | `01_hello_model.py` | invoke / stream / batch + 消息结构 |
| 01 | `02_first_agent.py` | 第一个 Agent，并行工具调用 |
| 02 | `01_model_init.py` | 三种创建方式 + 温度对比 + 模型名报错 |
| 02 | `02_thinking_mode.py` | 思考模式、reasoning_content、空答案坑 |
| 02 | `03_async_concurrency.py` | 串行 vs 并发（4.7x） vs 限流 |
| 03 | `01_messages.py` | 四种消息 + content_blocks + tool_call_id |
| 03 | `02_prompt_template.py` | 模板、LCEL、MessagesPlaceholder、partial |
| 03 | `03_trim_messages.py` | 上下文裁剪的代价 |
| 04 | `01_with_structured_output.py` | Pydantic / 枚举 / 嵌套 / include_raw |
| 04 | `02_agent_structured.py` | response_format + ToolStrategy 两个坑 |
| 04 | `03_extraction_pipeline.py` | 抽取流水线 + 业务校验 + 兜底 |
| 05 | `01_tool_basics.py` | 工具 schema + 手写 Agent 循环 |
| 05 | `02_tool_runtime.py` | ToolRuntime 读 context/state，权限下沉 |
| 05 | `03_tool_errors.py` | 三类失败 + 重试 + 兜底（含崩溃对比） |
| 06 | `01_agent_anatomy.py` | create_agent 全参数 + 完整轨迹 |
| 06 | `02_persistence.py` | checkpointer + thread 隔离 |
| 06 | `03_config_limits.py` | recursion_limit + GraphRecursionError |
| 07 | `01_custom_middleware.py` | 动态提示词 / 审计 / 工具拦截 |
| 07 | `02_builtin_middleware.py` | PII 脱敏 / 限流 / 模型降级 |
| 07 | `03_human_in_the_loop.py` | 审批：approve / reject / edit |
| 08 | `01_sqlite_persistence.py` | 跨进程记忆（`--read` 验证） |
| 08 | `02_long_term_memory.py` | Store 跨会话记忆 + 用户隔离 |
| 08 | `03_summarization.py` | 摘要压缩，关键信息保留验证 |
| 09 | `01_stream_modes.py` | updates/values/messages/custom 四模式 |
| 09 | `02_sse_and_events.py` | SSE 帧 + v3 事件协议 |
| 10 | `01_index_and_retrieve.py` | 加载/切分/向量化/检索/MMR |
| 10 | `02_rag_chain.py` | 完整 RAG + 检索前后对比 |
| 10 | `03_agentic_rag.py` | 检索工具化，自主决定搜不搜 |
| 11 | `01_agent_as_tool.py` | 主管-专家架构 + 跨领域汇总 |
| 11 | `02_graph_supervisor.py` | LangGraph 显式路由图 |
| 12 | `01_observability.py` | CallbackHandler 统计 token/耗时 |
| 12 | `02_eval.py` | 回归评测 + LLM-as-judge |
| 12 | `03_cost_and_cache.py` | 缓存命中（毫秒级）/ 成本估算表 |

---

## 依赖说明

```toml
"langchain>=1.4"                    # 核心：create_agent / middleware
"langchain-deepseek>=1.1.1"         # DeepSeek 集成
"langchain-text-splitters>=1.1.3"   # 文本切分
"langgraph-checkpoint-sqlite>=3.1.3" # 持久化
"python-dotenv"
"numpy"                             # 向量相似度计算
```

没有装 `langchain-classic`：v1 新代码不需要它，只在阅读旧项目时才装。

---

## 常见问题

**Q：能换成 OpenAI / Qwen / 本地 vLLM 吗？**
能。改 `src/langchain_learn/llm.py` 里的 `get_model()` 返回对应模型实例即可。注意两处差异：① 结构化输出的策略（OpenAI 支持 `ProviderStrategy`，其他多数要用 `ToolStrategy`）；② 思考模式的参数名（各厂商不同）。第 4 章有详细说明。

**Q：为什么示例都关掉了思考模式？**
两原因：思考模式不支持强制 `tool_choice`（结构化输出会 400），且会显著增加延迟与成本。需要推理能力的场景（第 2 章示例 2）再打开。

**Q：RAG 为什么用自制 embedding？**
DeepSeek 目前不提供 embedding 接口。为了不引入额外 API 依赖，示例用了确定性的字符 bigram 哈希向量，能跑通全流程但语义能力弱。生产环境换 `BAAI/bge-m3` 或 `text-embedding-3-small`（第 10 章有代码）。

**Q：跑示例花了多少钱？**
全部 33 个示例跑一遍约 0.1~0.2 元人民币（`deepseek-flash` 价格）。单次最贵的是第 11 章的多 Agent 示例。

**Q：Windows 上提示 `Event loop is closed`？**
见第 2 章 2.5 节：一个进程里只调用一次 `asyncio.run()`。

---

## 参考

- [LangChain v1 官方文档](https://docs.langchain.com/oss/python/langchain/overview)
- [What's new in LangChain v1](https://docs.langchain.com/oss/python/releases/langchain-v1)
- [LangGraph 文档](https://langchain-ai.github.io/langgraph/)
- [DeepSeek API 文档](https://api-docs.deepseek.com/)

---

版本基线：`langchain 1.4.7` / `langchain-core 1.7.3` / `langgraph 1.2.19` / `deepseek-flash`，最后更新于 2026 年 10 月。
