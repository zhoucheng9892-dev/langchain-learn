"""第 11 章 · 示例 1：把一个 Agent 当成另一个 Agent 的工具（最实用的多 Agent 形态）

运行：
    uv run python chapters/ch11_multiagent/01_agent_as_tool.py

为什么需要多 Agent？
    一个 Agent 塞 20 个工具、5 套规则，提示词会互相打架，工具选择准确率暴跌。
    拆成"主管 + 专家"之后，每个专家只面对自己领域的少数工具和一段短提示词。

最常见的三种协作方式：
    1. Agent as Tool（本例）：主管调用专家，专家不直接和用户对话；
    2. Handoff：主管把控制权整个交给专家，专家直接回复用户；
    3. 图编排（示例 2）：用 LangGraph 显式画出路由和回流。
"""

from langchain.agents import create_agent
from langchain.tools import tool

from langchain_learn import banner, get_model, section


# ---- 专家 A：数据查询 ---------------------------------------------------
@tool
def query_sales(region: str) -> str:
    """查询某大区的上月销售额（单位：万元）。"""
    data = {"华东": 1280, "华南": 960, "华北": 745, "西南": 512}
    return f"{region} 上月销售额 {data.get(region, 0)} 万元"


sales_agent = create_agent(
    model=get_model(max_tokens=384),
    tools=[query_sales],
    system_prompt="你是销售数据分析师，只回答销售数据相关问题，给出数字和一句解读，60 字以内。",
    name="sales_analyst",
)


# ---- 专家 B：制度问答 ---------------------------------------------------
POLICY = """
报销制度（2026 版）：
1. 差旅住宿：一线城市每晚不超过 600 元，其他城市不超过 400 元。
2. 市内交通：凭票实报实销，单日上限 100 元。
3. 招待费：需提前审批，单次不超过 2000 元。
4. 报销周期：每月 25 日前提交，次月 15 日到账。
"""
policy_agent = create_agent(
    model=get_model(max_tokens=384),
    tools=[],
    system_prompt=f"你是行政制度助手，只依据下面的制度回答，答不出来就说不知道。\n{POLICY}",
    name="policy_expert",
)


# ---- 把专家包装成工具 ---------------------------------------------------
@tool("ask_sales_analyst", description="向销售数据分析师提问，输入是完整的问题描述。")
def ask_sales(question: str) -> str:
    result = sales_agent.invoke({"messages": [{"role": "user", "content": question}]})
    return str(result["messages"][-1].content)


@tool("ask_policy_expert", description="向行政制度专家提问，输入是完整的问题描述。")
def ask_policy(question: str) -> str:
    result = policy_agent.invoke({"messages": [{"role": "user", "content": question}]})
    return str(result["messages"][-1].content)


def main() -> None:
    banner("1. 主管 Agent：自己不干活，只负责分派和汇总")

    supervisor = create_agent(
        model=get_model(max_tokens=512),
        tools=[ask_sales, ask_policy],
        system_prompt=(
            "你是团队主管。你不掌握具体数据，必须把问题转给对应专家：\n"
            "- 销售数字类 -> ask_sales_analyst\n"
            "- 制度规定类 -> ask_policy_expert\n"
            "如果一个问题同时涉及两者，分别提问后汇总。回答简洁。"
        ),
    )

    section("2. 单一领域问题：只调一个专家")
    r = supervisor.invoke({"messages": [{"role": "user", "content": "华东大区上月卖了多少？"}]})
    for m in r["messages"]:
        if getattr(m, "tool_calls", None):
            print(f"  -> 分派给 {[c['name'] for c in m.tool_calls]}")
    print(f"  主管汇总: {str(r['messages'][-1].content).strip()[:90]}")

    section("3. 跨领域问题：分派两次，再汇总")
    r2 = supervisor.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "我下个月要去上海出差 3 天，顺便拜访华东的客户。帮我看看华东业绩，以及住宿能报多少。",
                }
            ]
        },
        config={"recursion_limit": 12},
    )
    tool_names = [c["name"] for m in r2["messages"] if getattr(m, "tool_calls", None) for c in m.tool_calls]
    print(f"  分派记录: {tool_names}")
    print(f"  主管汇总: {str(r2['messages'][-1].content).strip()[:200]}")

    section("4. 这种架构的取舍")
    print("  优点：每个专家上下文短、工具少，选择准确率高；提示词可独立迭代。")
    print("  缺点：多一层调用 = 更多 token 与延迟；主管可能分派错。")
    print("  替代方案：把专家写成 skill，由单 Agent 按需加载（见第 12 章）。")


if __name__ == "__main__":
    main()
