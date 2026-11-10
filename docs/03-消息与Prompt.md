# 第 3 章 消息与 Prompt：对话的底层数据结构

> 本章代码：`chapters/ch03_messages/01_messages.py`、`02_prompt_template.py`、`03_trim_messages.py`

## 3.1 本章目标

- 分清四种消息类型，知道每种由谁来"填"
- 会用 `content_blocks` 读取跨供应商的统一内容块
- 会用 `ChatPromptTemplate` 和 `trim_messages` 管理上下文

## 3.2 四种消息

| 类型 | 谁生成 | 关键字段 | 说明 |
| --- | --- | --- | --- |
| `SystemMessage` | 开发者 | `content` | 角色与规则，放在最前 |
| `HumanMessage` | 用户 | `content` | 用户输入 |
| `AIMessage` | 模型 | `content`、`tool_calls` | 可能带工具调用请求 |
| `ToolMessage` | 框架 | `content`、`tool_call_id` | 必须与某次调用 id 对应 |

Agent 的一次完整执行长这样：

```
HumanMessage              "北京天气？"
AIMessage(tool_calls=[…]) 模型决定调 get_weather
ToolMessage               "北京: 晴, 25°C"      <- tool_call_id 对上前一个
AIMessage                 "北京现在晴，25°C。"
```

`tool_call_id` 对不上会直接报错。手写工具循环时（第 5 章）要注意这一点；用 `create_agent` 时框架帮你填。

## 3.3 content_blocks：v1 的新东西

`.content` 在纯文本场景是字符串，但多模态场景会变成列表。v1 提供统一访问方式：

```python
reply = model.invoke("你好")
reply.content_blocks
# [{'type': 'text', 'text': '你好！有什么可以帮你的吗？'}]
```

标准内容块类型包括 `text`、`image`、`reasoning`、`tool_call`、`citation` 等。好处是换供应商不用改解析代码——OpenAI 的 `image_url`、Anthropic 的 `image` 都会被规整成同一种结构。

**实践建议**：新代码优先读 `content_blocks`，需要兼容老代码时再退回 `.content`。

## 3.4 多轮对话：模型无状态，列表有状态

```python
history = [SystemMessage("你是一个简洁的助手。")]
for q in ["我叫小林", "我叫什么？"]:
    history.append(HumanMessage(q))
    history.append(model.invoke(history))
```

每次调用都把**完整历史**发过去。所以：

- 历史越长，token 越多，越贵越慢；
- 历史有上限，超了就报错或被截断；
- "记忆"这件事，本质上是"你把多少历史塞回去"。

这就是第 8 章 checkpointer 要解决的问题——它帮你自动存取这个列表。

## 3.5 Prompt 模板与 LCEL 管道

`|` 是 LCEL 的语法糖，本质是 `RunnableSequence`，天生支持 `invoke / stream / batch / ainvoke`。

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是{company}的客服，回答不超过 40 字。"),
    ("human", "{question}"),
])
chain = prompt | model | StrOutputParser()

chain.invoke({"company": "星河云", "question": "支持退款吗？"})
chain.stream({...})    # 流式
chain.batch([{...}, {...}])  # 批量（自动并发）
```

三个常用技巧：

| 技巧 | 写法 | 用途 |
| --- | --- | --- |
| 插入历史 | `MessagesPlaceholder("history")` | 多轮对话模板 |
| 预设变量 | `prompt.partial(brand="X", tone="Y")` | 固定品牌/语气 |
| 先看效果 | `prompt.invoke({...})` | 打印最终发给模型的消息 |

`prompt.invoke()` 返回的是 `PromptValue`，可以 `.messages` 查看消息列表——调试提示词时这一步非常有用，比猜"模型看到了什么"靠谱。

## 3.6 上下文裁剪 trim_messages

```python
from langchain.messages import trim_messages
from langchain_core.messages.utils import count_tokens_approximately

trimmed = trim_messages(
    history,
    max_tokens=120,
    strategy="last",              # 从最新往回保留
    token_counter=count_tokens_approximately,
    include_system=True,          # system 永不丢
    allow_partial=False,          # 不切半条消息
    start_on="human",             # 裁剪后第一条必须是 human
)
```

`start_on="human"` 很关键：如果裁完之后第一条是 `AIMessage`，部分供应商会报错或行为异常。

实测（`03_trim_messages.py`）：26 条消息 / 246 token → 裁到 12 条 / 113 token。代价是模型看不到前 5 轮，回答开始出错（"你问了 5 次"而实际是 12 次）。

**丢历史是有代价的**。要既省 token 又不丢信息，用第 7、8 章的 `SummarizationMiddleware`——它把老消息压成摘要，而不是直接删。

## 3.7 常见坑

| 现象 | 原因 | 解决 |
| --- | --- | --- |
| `tool_call_id` 不匹配 | 手工构造 ToolMessage 时漏填 | 从 `tool_calls[i]["id"]` 复制 |
| 模板变量报 KeyError | 漏传变量 | `prompt.input_variables` 先查清 |
| 裁剪后模型答错 | 关键信息被裁掉 | 用摘要，或提高 `max_tokens` |
| 流式时 `content` 是空 | 思考模式只有 reasoning 增量 | 判断两个字段都有无内容 |

## 3.8 练习

1. 用 `ChatPromptTemplate` 写一个"翻译 + 语气调整"的双变量模板，用 `partial` 固定目标语言。
2. 构造 30 轮假对话，用 `trim_messages` 裁到 100 token，打印第一条消息类型验证 `start_on` 生效。
3. 用 `content_blocks` 打印一次带工具调用的 `AIMessage`，找出工具调用对应哪个块类型。

下一章：让模型稳定输出结构化数据——JSON 解析的时代结束了。
