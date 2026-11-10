"""第 11 章 · 示例 2：用 LangGraph 显式编排多 Agent（supervisor 图）

运行：
    uv run python chapters/ch11_multiagent/02_graph_supervisor.py

Agent-as-Tool 的问题是"黑盒"：路由藏在模型的一次工具调用里，
你无法强制顺序、无法加人工审核、无法做条件回流。

需要确定性时，就把流程画成图：
    START -> supervisor -> (条件边) -> math_expert / doc_expert -> supervisor -> ... -> END
这就是 v1 底层跑的那套东西，自己画图反而更可控。
"""

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from langchain.agents import create_agent
from langchain.tools import tool

from langchain_learn import banner, get_model, section


# ---- 图的状态 -----------------------------------------------------------
class State(TypedDict):
    messages: Annotated[list, add_messages]  # add_messages：追加而不是覆盖
    route: str                                # 当前路由结果
    turns: int                                # 已分派轮次（防止无限回流）


# ---- 专家节点 -----------------------------------------------------------
@tool
def calc(expression: str) -> str:
    """计算四则运算表达式。"""
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as exc:  # noqa: BLE001
        return f"计算失败：{exc}"


math_agent = create_agent(
    model=get_model(max_tokens=384),
    tools=[calc],
    system_prompt="你是数学专家，负责精确计算，回答里必须给出算式和结果。",
)

doc_agent = create_agent(
    model=get_model(max_tokens=384),
    tools=[],
    system_prompt="你是文档专家，擅长把结论写成结构清晰的三行要点。",
)


def math_node(state: State) -> dict:
    result = math_agent.invoke({"messages": state["messages"]})
    return {
        "messages": [AIMessage(content=f"[数学专家] {result['messages'][-1].content}", name="math")],
        "turns": state.get("turns", 0) + 1,
    }


def doc_node(state: State) -> dict:
    result = doc_agent.invoke({"messages": state["messages"]})
    return {
        "messages": [AIMessage(content=f"[文档专家] {result['messages'][-1].content}", name="doc")],
        "turns": state.get("turns", 0) + 1,
    }


# ---- 主管节点：只做路由决策，不生成内容 --------------------------------
class Route(BaseModel):
    next: Literal["math", "doc", "finish"] = Field(description="下一个节点")
    reason: str = Field(description="一句话说明理由")


def supervisor_node(state: State) -> dict:
    turns = state.get("turns", 0)
    if turns >= 3:  # 硬性上限：回流超过 3 次强制收尾
        return {"route": "finish"}

    done = [m.name for m in state["messages"] if getattr(m, "name", None)]
    router = get_model(max_tokens=256).with_structured_output(Route)
    decision = router.invoke(
        [
            *state["messages"],
            HumanMessage(
                "你是任务调度员。判断下一步该谁处理。\n"
                f"已完成的节点：{done or ['无']}\n"
                "规则：\n"
                "- 用户需要算术，且 math 还没做过 -> math\n"
                "- math 已做过，且用户还要求整理成文档/简报/要点，而 doc 还没做过 -> doc\n"
                "- 其余情况 -> finish\n"
                "只输出路由决定。"
            ),
        ]
    )
    print(f"    [supervisor] -> {decision.next}（{decision.reason}）")
    return {"route": decision.next}


def route_fn(state: State) -> str:
    return state.get("route", "finish")


def main() -> None:
    builder = StateGraph(State)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("math", math_node)
    builder.add_node("doc", doc_node)

    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
        "supervisor", route_fn, {"math": "math", "doc": "doc", "finish": END}
    )
    builder.add_edge("math", "supervisor")  # 专家干完活回到主管，形成回流
    builder.add_edge("doc", "supervisor")

    graph = builder.compile()

    banner("1. 需要计算的请求：supervisor -> math -> supervisor -> finish")
    out = graph.invoke(
        {
            "messages": [
                HumanMessage("我们团队 7 个人，人均预算 480 元，加上 15% 的服务费，总共要多少钱？")
            ],
            "turns": 0,
        },
        config={"recursion_limit": 20},
    )
    for m in out["messages"]:
        tag = getattr(m, "name", None) or type(m).__name__
        print(f"  [{tag}] {str(m.content)[:100].replace(chr(10), ' ')}")

    section("2. 需要两步的请求：先算，再整理成文档")
    out2 = graph.invoke(
        {
            "messages": [
                HumanMessage("算一下 1280*0.85 是多少，然后把结果整理成三行要点给我。")
            ],
            "turns": 0,
        },
        config={"recursion_limit": 25},
    )
    print(f"  共产生 {len(out2['messages'])} 条消息，回流 {out2['turns']} 次")
    print(f"  最终回答:\n{str(out2['messages'][-1].content)[:220]}")

    section("3. 图的结构（节点与边）")
    drawable = graph.get_graph()
    print(f"  节点: {list(drawable.nodes)}")
    for edge in drawable.edges:
        print(f"  边  : {edge.source} -> {edge.target}" + (f"  (条件: {edge.data})" if edge.data else ""))
    print("  想看图形版：pip install grandalf && graph.get_graph().draw_ascii()")

    section("4. 什么时候画图，什么时候用 Agent-as-Tool")
    print("  流程确定、需要强制顺序/审核/回流  -> 画图（本例）")
    print("  流程开放、由模型自由判断          -> create_agent + 专家工具")
    print("  两者可以混用：图里放 Agent 节点，Agent 里再挂工具。")


if __name__ == "__main__":
    main()
