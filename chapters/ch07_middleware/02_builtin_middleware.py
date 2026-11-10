"""第 7 章 · 示例 2：内置中间件 —— 不用自己写，开箱即用

运行：
    uv run python chapters/ch07_middleware/02_builtin_middleware.py

v1 内置了一批"人人都会需要"的中间件，直接往 middleware=[] 里加即可：
    PIIMiddleware          输入/输出脱敏
    ToolCallLimitMiddleware 限制工具调用次数
    ModelCallLimitMiddleware 限制模型调用次数
    ModelFallbackMiddleware 主模型挂了自动切备用模型
    ModelRetryMiddleware   模型调用失败重试
    SummarizationMiddleware 历史过长时自动摘要（第 8 章）
    HumanInTheLoopMiddleware 敏感操作人工审批（示例 3）
"""

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelFallbackMiddleware,
    PIIMiddleware,
    ToolCallLimitMiddleware,
)
from langchain.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain.tools import tool

from langchain_learn import MODEL_REF, banner, get_model, section

TOOL_CALLS = {"count": 0}


@tool
def send_email(to: str, body: str) -> str:
    """给指定邮箱发送邮件。"""
    TOOL_CALLS["count"] += 1
    return f"已发送给 {to}"


@tool
def search_contact(name: str) -> str:
    """按姓名查联系人邮箱。"""
    return f"{name} 的邮箱是 zhangsan@example.com"


def main() -> None:
    banner("1. PIIMiddleware：邮箱在进入模型前就被脱敏")

    pii_agent = create_agent(
        model=get_model(max_tokens=256),
        tools=[],
        system_prompt="你是助手，原样复述用户给的信息。",
        middleware=[PIIMiddleware("email", strategy="redact", apply_to_input=True)],
    )
    r = pii_agent.invoke(
        {"messages": [HumanMessage("请复述这句话：请联系 zhangsan@example.com 处理退款。")]}
    )
    print(f"  模型看到的已经是脱敏后的文本，所以它回答：{r['messages'][-1].content.strip()[:70]}")
    print("  注意：脱敏发生在 middleware 层，和提示词无关，模型无法绕过。")

    section("2. ToolCallLimitMiddleware：给工具调用上保险")

    # thread_limit 按整个会话计数；run_limit 按单次 invoke 计数。
    # 这里限制"整个会话最多发 1 封邮件"，第二次会被拦下。
    limited_agent = create_agent(
        model=get_model(max_tokens=512),
        tools=[search_contact, send_email],
        system_prompt="你是助理，用户让你发邮件就直接发。",
        checkpointer=InMemorySaver(),
        middleware=[ToolCallLimitMiddleware(tool_name="send_email", thread_limit=1)],
    )
    cfg = {"configurable": {"thread_id": "mail-demo"}}

    r1 = limited_agent.invoke(
        {"messages": [HumanMessage("给张三发一封会议提醒邮件。")]}, config=cfg
    )
    print(f"  第 1 次: 累计发送 {TOOL_CALLS['count']} 封 -> {r1['messages'][-1].content.strip()[:50]}")

    r2 = limited_agent.invoke(
        {"messages": [HumanMessage("再给张三发一封感谢邮件。")]}, config=cfg
    )
    print(f"  第 2 次: 累计发送 {TOOL_CALLS['count']} 封（被拦）")
    print(f"  模型收到的提示: {str(r2['messages'][-1].content).strip()[:80]}")

    section("3. ModelFallbackMiddleware：主模型不可用时自动降级")

    # 故意把主模型写成不存在的名字，模拟"某个模型下线了"
    fallback_agent = create_agent(
        model="deepseek:deepseek-v4.1-flash",  # 会 400
        tools=[],
        system_prompt="回答一句话。",
        middleware=[ModelFallbackMiddleware(MODEL_REF)],  # 备用模型
    )
    r = fallback_agent.invoke({"messages": [HumanMessage("说一句欢迎语。")]})
    print(f"  主模型报错后自动切到备用模型，回答：{r['messages'][-1].content.strip()[:60]}")

    section("4. 中间件叠加顺序")
    print("  middleware=[A, B, C] 时：")
    print("    before_* 按 A -> B -> C 顺序执行")
    print("    after_*  按 C -> B -> A 逆序执行（洋葱模型）")
    print("  所以：限流/鉴权放前面，脱敏放中间，日志放最后。")


if __name__ == "__main__":
    main()
