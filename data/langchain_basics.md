# LangChain 1.x 核心概念速览

LangChain 1.0 于 2025 年 10 月 22 日发布，与 LangGraph 1.0 同日。这次大版本的核心动作是把框架收敛成"智能体构建基础"：一个 `create_agent` 函数加上一套中间件机制，取代了过去的 Chain、Memory、AgentExecutor 等一整套抽象。

## 四层抽象

1. **模型层（ChatModel）**：统一各供应商的聊天接口。`init_chat_model("provider:model")` 用字符串引用模型，`ChatDeepSeek`、`ChatOpenAI` 等具体类提供完整参数。
2. **消息层（Messages）**：`SystemMessage`、`HumanMessage`、`AIMessage`、`ToolMessage` 四种消息构成对话历史。v1 新增 `content_blocks` 属性，把文本、图片、推理过程统一成标准内容块。
3. **工具层（Tools）**：`@tool` 装饰器把普通函数变成带 JSON Schema 的可调用对象。工具可以通过 `ToolRuntime` 读取会话状态与调用上下文。
4. **智能体层（Agent）**：`create_agent(model, tools, system_prompt)` 返回一个可执行图，内部循环是"调用模型 -> 执行工具 -> 再调用模型"，直到模型不再请求工具。

## 包拆分

`langchain` 包只保留智能体必需的核心构件：`langchain.agents`、`langchain.chat_models`、`langchain.messages`、`langchain.tools`、`langchain.embeddings`。历史遗留的 Chain、Retriever、Indexing API、Hub 全部迁移到独立的 `langchain-classic` 包，需要手动安装。供应商集成放在各自的包中，例如 `langchain-deepseek`、`langchain-openai`。

## 与 0.x 的关键差异

| 0.x 写法 | 1.x 写法 |
| --- | --- |
| `AgentExecutor` + `create_react_agent` | `create_agent` |
| `prompt=` 参数 | `system_prompt=` 参数 |
| `pre_model_hook` / `post_model_hook` | `before_model` / `after_model` 中间件 |
| `LLMChain` | `prompt \| model \| parser` |
| `langchain.retrievers` | `langchain_classic.retrievers` |

## 日常使用建议

新项目一律从 `create_agent` 起步，只有确实需要固定流程编排时才退回 LCEL 管道或 LangGraph 显式图。能用中间件解决的问题不要写进提示词，因为提示词不可靠，代码才可靠。
