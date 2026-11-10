# 第 10 章 RAG：让模型回答它没见过的东西

> 本章代码：`chapters/ch10_rag/01_index_and_retrieve.py`、`02_rag_chain.py`、`03_agentic_rag.py`

## 10.1 本章目标

- 跑通 加载 → 切分 → 向量化 → 存储 → 检索 全流程
- 写出一个带来源标注的 RAG 问答
- 分清"固定管道"和"Agentic RAG"的取舍

## 10.2 RAG 的本质

模型的知识截止在训练时间，且不知道你的内部文档。RAG 就是：**先把相关资料找出来，塞进提示词，再让模型基于资料回答**。

```
问题 -> 检索 -> 相关片段 -> [片段 + 问题] -> 模型 -> 答案
```

它解决三个问题：知识时效、私有知识、可溯源（答案能标来源）。

## 10.3 五个环节

| 环节 | 做什么 | 本项目实现 |
| --- | --- | --- |
| Load | 读文件 | `data/*.md` |
| Split | 切成合适大小的片段 | `MarkdownHeaderTextSplitter` + `RecursiveCharacterTextSplitter` |
| Embed | 文本转向量 | `HashingEmbeddings`（教学用） |
| Store | 存入向量库 | `InMemoryVectorStore` |
| Retrieve | 相似度检索 | `similarity_search_with_score` / MMR |

### 切分策略：比模型更影响效果

```python
# 两级切分：先按标题，再按长度兜底
header_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "h1"), ("##", "h2")],
    strip_headers=False,      # 保留标题，标题本身就是强信号
)
length_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400, chunk_overlap=60, separators=["\n\n", "\n", "。", "，", " "]
)
```

| 参数 | 太小的后果 | 太大的后果 |
| --- | --- | --- |
| `chunk_size` | 语义被切碎，检索到半句话 | 混入大量无关内容，精准度下降 |
| `chunk_overlap` | 跨段落的语义断裂 | 冗余，浪费 token |

中文场景常用 `chunk_size=300~500`、`overlap=50~100`。**按标题/段落切优先于按字数切。**

### 关于本项目的 Embedding

DeepSeek 目前不提供 embedding 接口，所以示例用了一个自实现的确定性哈希向量（`src/langchain_learn/embeddings.py`）：字符 bigram 哈希到 512 维再归一化。

它的语义能力远不如真实模型，**只能看字面重合**，所以你会看到"与 0.x 的关键差异"排在"六个钩子"前面——因为前者也包含了 `before_model` 这些词。这正是教学目的：**流程可以离线跑通，向量模型随时可换**。

生产环境换成：

```python
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")   # 中文效果好、可本地部署

from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
```

## 10.4 固定管道 RAG

```python
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "只能依据 <context> 回答；资料没有就说「资料中未提及」；"
     "末尾用「来源：文件名」标注；回答 150 字以内。"),
    ("human", "<context>\n{context}\n</context>\n\n问题：{question}"),
])

chain = RAG_PROMPT | model | StrOutputParser()
answer = chain.invoke({"context": format_docs(retriever.invoke(q)), "question": q})
```

提示词里那三条规则不是装饰：

| 规则 | 不写会怎样 |
| --- | --- |
| 只能依据 context | 模型混合自己的记忆，答案无法溯源 |
| 没有就说不知道 | 编造（比不检索更危险，因为它看起来有依据） |
| 标注来源 | 用户无法验证，排查幻觉也失去线索 |

实测（`02_rag_chain.py`）对比：

```
无检索: "1.x 里 create_agent 取代了..."（内容大致对，但无法确认依据）
有检索: 检索到 [langchain_basics.md > 与 0.x 的关键差异] -> 准确 + 标了来源
无答案: "资料中未提及 LangChain 默认支持哪些向量数据库"  <- 诚实拒答
```

## 10.5 检索质量优化手段

| 手段 | 做法 | 效果 |
| --- | --- | --- |
| 提高召回 | `k` 从 3 调到 8，再重排 | 减少漏检 |
| MMR | `search_type="mmr"` | 结果多样，避免同段落重复 |
| 阈值过滤 | 分数低于阈值直接丢 | 减少噪声 |
| 混合检索 | 向量 + 关键词（BM25） | 专有名词命中率显著提升 |
| 重排 | 用 cross-encoder 对 top-20 重排 | 精度提升最明显，但更慢 |
| 元数据过滤 | 按文档类型/时间过滤 | 缩小搜索空间 |

```python
vector_store.as_retriever(
    search_type="mmr", search_kwargs={"k": 2, "fetch_k": 6, "lambda_mult": 0.5}
)
```

`lambda_mult` 越小越强调多样性，越大越强调相关性。

## 10.6 Agentic RAG

把检索做成工具，让模型自己决定搜不搜、怎么搜：

```python
@tool
def search_docs(query: str) -> str:
    """在内部技术文档中检索资料。query 用关键词，不要用整句疑问句。"""
    ...
```

对比：

| | 固定管道 | Agentic RAG |
| --- | --- | --- |
| 调用次数 | 1 次检索 + 1 次生成 | N 次检索 + M 次生成 |
| 优点 | 稳定、便宜、延迟可预测 | 能判断"不需要查"、能改写查询、能多跳 |
| 缺点 | 无论什么问题都检索一次 | 成本延迟上升，可能绕圈 |

实测（`03_agentic_rag.py`）：

```
[1] "SummarizationMiddleware 的 trigger 支持哪几种阈值？" -> 触发检索，答对
[2] "你好，随便聊两句"                                   -> 不检索，直接回答（省钱）
[3] "哪个中间件能防止 Agent 停不下来？怎么配？"          -> 并行检索两次，综合回答
```

**建议：先做固定管道，只有确实需要多跳/条件检索时才升级。**

## 10.7 常见坑

| 现象 | 原因 | 解决 |
| --- | --- | --- |
| 检索到了但还是答错 | 片段顺序/拼接方式差 | 加片段编号，标注来源 |
| 答得头头是道但全是编的 | 提示词没限制"只能用资料" | 明确"资料没有就说不知道" |
| 检索不到 | 切分太大/太小，或查询词与文档词汇不匹配 | 调切分 + 混合检索 + 让模型改写查询 |
| 专有名词搜不到 | 纯向量检索对罕见词不敏感 | 加 BM25 混合检索 |
| 上下文太长 | k 太大 | k=3~5，配合重排 |

## 10.8 练习

1. 把 `chunk_size` 从 400 改成 800 和 150，各跑一次 `01_index_and_retrieve.py`，比较检索结果。
2. 给 `RAG_PROMPT` 加一条"如果片段之间矛盾，指出矛盾"，构造一个矛盾场景测试。
3. 把 `build_retriever` 的 `k` 改成 6，看 `02_rag_chain.py` 的答案质量与 token 变化。
4. 给 `search_docs` 加一个 `top_k` 参数，让 Agent 自己决定搜几条。

下一章：多 Agent。当一个 Agent 的工具超过 10 个，就该拆了。
