"""第 5 章 · 示例 2：让工具拿到"运行时信息"（ToolRuntime）

运行：
    uv run python chapters/ch05_tools/02_tool_runtime.py

问题：工具怎么知道"当前是哪个用户"、"这是第几轮对话"、"用户买了什么套餐"？
把 user_id 写在工具参数里？那模型就能随便填别人的 id，既不安全也不可靠。

v1 的答案：ToolRuntime。它由框架注入，不暴露给模型，
工具里可以直接读 state（会话状态）、context（每次调用传入的上下文）、store（长期记忆）。
"""

from dataclasses import dataclass

from langchain.agents import AgentState, create_agent
from langchain.tools import ToolRuntime, tool

from langchain_learn import banner, get_model, section


# ---- 1. 每次调用传入的上下文（不进入模型上下文，安全） ------------------
@dataclass
class UserContext:
    user_id: str
    plan: str  # free / pro


# ---- 2. 自定义 Agent 状态：可以跨轮次累计 ------------------------------
class MyState(AgentState):
    queries: int  # 已经问过几个问题


@tool
def who_am_i(runtime: ToolRuntime[UserContext]) -> str:
    """查询当前用户的身份与套餐。"""
    user = runtime.context
    return f"user_id={user.user_id}, plan={user.plan}"


@tool
def count_my_questions(runtime: ToolRuntime[UserContext]) -> str:
    """查询本次会话中用户已经问了多少个问题。"""
    # state 里是完整消息列表，可以自己数
    messages = runtime.state["messages"]
    human_count = sum(1 for m in messages if type(m).__name__ == "HumanMessage")
    return f"{runtime.context.user_id} 本次会话已提问 {human_count} 次"


@tool
def premium_only_feature(runtime: ToolRuntime[UserContext]) -> str:
    """一个只有 pro 套餐才能用的高级功能。"""
    if runtime.context.plan != "pro":
        # 工具有权拒绝：返回错误信息比抛异常更温和，模型会转述给用户
        return "拒绝：该功能仅对 pro 套餐开放。"
    return f"已为 {runtime.context.user_id} 生成高级分析报告（演示数据）。"


def main() -> None:
    banner("1. 工具读取 context：用户身份与套餐")

    agent = create_agent(
        model=get_model(max_tokens=512),
        tools=[who_am_i, count_my_questions, premium_only_feature],
        system_prompt="你是账户助手，涉及用户信息必须调用工具，不要猜测。",
        context_schema=UserContext,  # 声明 context 的结构
        state_schema=MyState,        # 声明扩展状态
    )

    # context 在每次 invoke 时传入，和 config/thread_id 是两回事
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "我是谁？我的套餐是什么？我已经问了几个问题？"}]},
        context=UserContext(user_id="u_10086", plan="free"),
    )
    print("  free 用户：")
    print(f"    {result['messages'][-1].content.strip()}")

    section("2. 同一个 Agent，换一个 context，工具行为立刻不同")
    result2 = agent.invoke(
        {"messages": [{"role": "user", "content": "帮我用一下高级分析功能。"}]},
        context=UserContext(user_id="u_9527", plan="pro"),
    )
    print("  pro 用户：")
    print(f"    {result2['messages'][-1].content.strip()}")

    section("3. 谁看得见 runtime？")
    print("  模型只看到 who_am_i() 这个空参数函数——它无法伪造 user_id。")
    print("  runtime 里的 state/context/store 由框架在调用瞬间注入。")
    print("  这正是把'权限校验'放在工具层而不是提示词层的原因。")


if __name__ == "__main__":
    main()
