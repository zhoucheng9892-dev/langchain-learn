"""第 6 章 · 示例 3：运行时配置与步数上限（recursion_limit）

运行：
    uv run python chapters/ch06_agent/03_config_limits.py

Agent 的危险不在于它笨，而在于它可能"停不下来"：
工具一直失败、模型反复重试，账单就上去了。
recursion_limit 是硬闸门——按"超步（superstep）"计数，次数就是安全边际。
"""

from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.errors import GraphRecursionError

from langchain_learn import banner, get_model, section


@tool
def slow_lookup(key: str) -> str:
    """查询某个配置项的值。"""
    return f"{key} = 42"


@tool
def another_lookup(key: str) -> str:
    """从另一张表查询同一个配置项。"""
    return f"{key} = 43"


def main() -> None:
    agent = create_agent(
        model=get_model(max_tokens=512),
        tools=[slow_lookup, another_lookup],
        system_prompt="你是运维助手。需要交叉核对时把两个查询工具都用上。",
    )

    question = "帮我把 timeout 这个配置项在两个来源都查一下，并交叉核对。"

    banner("1. 正常执行（默认 recursion_limit=25）")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config={
            "recursion_limit": 25,
            "tags": ["demo", "ch06"],           # 用于追踪系统里筛选
            "metadata": {"user_id": "u_1"},      # 任意业务标签
            "run_name": "config-demo",           # 在 LangSmith 里显示的名字
        },
    )
    print(f"  消息数: {len(result['messages'])}")
    print(f"  回答  : {result['messages'][-1].content.strip()[:80]}")

    section("2. 把上限调成 3：观察 GraphRecursionError")
    try:
        agent.invoke(
            {"messages": [{"role": "user", "content": question}]},
            config={"recursion_limit": 3},
        )
    except GraphRecursionError as exc:
        print(f"  {type(exc).__name__}: {str(exc).splitlines()[0][:120]}")
        print("  说明：模型 -> 工具 -> 模型 至少要 3 个超步，加上收尾就超了。")

    section("3. 生产建议")
    print("  - 永远显式设 recursion_limit，常见 10~30，按任务复杂度调；")
    print("  - 捕获 GraphRecursionError 后给用户一个兜底话术，而不是 500；")
    print("  - 配合第 7 章的 ToolCallLimitMiddleware / ModelCallLimitMiddleware 做更细的限流。")

    section("4. 兜底写法")
    def safe_run(text: str) -> str:
        try:
            out = agent.invoke(
                {"messages": [{"role": "user", "content": text}]},
                config={"recursion_limit": 12},
            )
            return str(out["messages"][-1].content)
        except GraphRecursionError:
            return "这个问题比较复杂，我先整理一下已有信息，稍后给你完整答复。"

    print(f"  {safe_run('现在几点了？')[:60]}")


if __name__ == "__main__":
    main()
