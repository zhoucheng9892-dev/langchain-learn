# 第 11 章 多 Agent：拆开是为了让每个 Agent 更笨

> 本章代码：`chapters/ch11_multiagent/01_agent_as_tool.py`、`02_graph_supervisor.py`

## 11.1 本章目标

- 知道什么时候该拆多 Agent（以及什么时候不该）
- 会用 "Agent as Tool" 做主管-专家架构
- 会用 LangGraph 画显式的路由图

## 11.2 为什么要拆

一个 Agent 塞 20 个工具、5 套规则会发生什么：

- 工具选择准确率暴跌（模型要在 20 个里挑一个）
- 提示词互相打架（"你必须严谨" vs "你要幽默"）
- 出问题无法定位（是路由错了、工具错了、还是提示词错了）

拆成"主管 + 专家"之后，每个专家只面对 3 个工具和一段短提示词。**代价是多一层调用。**

判断标准：

| 信号 | 该拆 | 不该拆 |
| --- | --- | --- |
| 工具数量 | > 10 | ≤ 6 |
| 领域差异 | 销售/法务/运维，规则完全不同 | 同一领域的不同操作 |
| 权限边界 | 不同角色能碰的数据不同 | 权限一致 |
| 上下文长度 | 单 Agent 提示词 > 2000 token | 短提示词够用 |

## 11.3 三种协作形态

| 形态 | 控制流 | 适用 |
| --- | --- | --- |
| Agent as Tool | 主管调专家，专家不直接回用户 | 最常见、最好调试 |
| Handoff | 主管把控制权整体移交专家 | 客服转人工、多轮专属对话 |
| 图编排 | 显式节点 + 条件边 | 需要强制顺序、审核、回流 |

## 11.4 Agent as Tool

```python
sales_agent = create_agent(
    model=get_model(), tools=[query_sales],
    system_prompt="你是销售数据分析师，只回答销售数据问题。",
    name="sales_analyst",
)

@tool("ask_sales_analyst", description="向销售数据分析师提问，输入是完整的问题描述。")
def ask_sales(question: str) -> str:
    result = sales_agent.invoke({"messages": [{"role": "user", "content": question}]})
    return str(result["messages"][-1].content)

supervisor = create_agent(
    model=get_model(), tools=[ask_sales, ask_policy],
    system_prompt="你是主管。销售数字类转 ask_sales_analyst，制度类转 ask_policy_expert。",
)
```

实测（`01_agent_as_tool.py`）：

```
[2] "华东大区上月卖了多少？"        -> 分派 ask_sales_analyst -> "1280 万元"
[3] "去上海出差，看华东业绩 + 住宿标准" -> 分派两次，主管汇总两段结果
```

注意第 3 步：跨领域问题被拆成两次子调用，主管负责合并。这是多 Agent 最有价值的能力——**分工 + 汇总**。

三个要点：

1. 专家的 `system_prompt` 要**只描述自己的领域**，越窄越准；
2. 包装工具的 `description` 要写清"什么时候该用它"，这是主管的路由依据；
3. 专家内部不要再挂"主管"，避免无限递归。

## 11.5 图编排：把流程画出来

Agent-as-Tool 的路由藏在模型的一次工具调用里，你无法强制顺序。需要确定性时画图：

```python
class State(TypedDict):
    messages: Annotated[list, add_messages]
    route: str
    turns: int

builder = StateGraph(State)
builder.add_node("supervisor", supervisor_node)
builder.add_node("math", math_node)
builder.add_node("doc", doc_node)

builder.add_edge(START, "supervisor")
builder.add_conditional_edges("supervisor", route_fn, {"math": "math", "doc": "doc", "finish": END})
builder.add_edge("math", "supervisor")   # 干完活回到主管，形成回流
builder.add_edge("doc", "supervisor")
graph = builder.compile()
```

关键设计：

| 设计 | 作用 |
| --- | --- |
| `add_messages` reducer | 消息追加而不是覆盖 |
| `turns` 计数 | 回流超过 3 次强制收尾，防死循环 |
| `route` 字段 | 条件边的判断依据 |
| supervisor 只做路由 | 不让它生成内容，职责单一 |

实测（`02_graph_supervisor.py`）：

```
[supervisor] -> math（用户需要计算，math 尚未执行）
[supervisor] -> doc（math 已完成，用户还要求整理成要点）
[supervisor] -> finish（math 与 doc 均已完成）

图结构：
  节点: ['__start__', 'supervisor', 'math', 'doc', '__end__']
  边  : __start__ -> supervisor
  边  : math -> supervisor / doc -> supervisor
  边  : supervisor -> math (条件) / -> doc (条件) / -> __end__ (条件: finish)
```

**路由节点不要复用对话历史做隐式判断**，把"已完成哪些节点"显式传给它（本例的 `done` 列表），路由准确率会高很多。

## 11.6 什么时候别拆

多 Agent 不是高级，是**成本**：每多一层就多一次模型调用、多一份 token、多一个故障点。

以下情况单 Agent 就够：

- 工具 ≤ 6 个，且都是同一领域
- 流程线性，不需要路由
- 团队还没有可观测手段（多 Agent 出问题很难查）

一个更轻的替代方案：**把专家写成 skill，由单 Agent 按需加载**。这也是 Deep Agents 的思路——用文件系统 + 按需读文档，而不是拆 Agent。

## 11.7 常见坑

| 现象 | 原因 | 解决 |
| --- | --- | --- |
| 主管不分派，自己瞎答 | 工具描述太模糊 | description 写清使用场景 |
| 无限循环 | 回流没有上限 | 计数器 + `recursion_limit` |
| 专家答非所问 | 专家提示词太宽 | 只描述本领域，明确拒答边界 |
| 汇总丢信息 | 主管只转述最后一句 | 让专家输出结构化要点 |
| 成本翻倍 | 拆得太细 | 合并同领域专家 |

## 11.8 练习

1. 给 `01_agent_as_tool.py` 加第三个专家（如"法务"），观察主管路由是否还准。
2. 把 `02_graph_supervisor.py` 的 `turns` 上限改成 1，观察强制收尾的效果。
3. 给 supervisor 节点加 `response_format`，把路由决策做成结构化输出而不是 `Literal` 字段。
4. 把两个专家合并成一个 Agent，对比工具选择准确率与 token 消耗。

下一章：工程化。能跑只是开始，能上线才算学会。
