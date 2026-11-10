"""第 7 章 · 示例 3：Human-in-the-loop —— 让敏感操作等人点头

运行：
    uv run python chapters/ch07_middleware/03_human_in_the_loop.py

Agent 一旦能"做事"，就必须能"被拦住"。
v1 的做法：HumanInTheLoopMiddleware 在工具执行前触发 interrupt，
调用方拿到中断信息 -> 人去审批 -> 用 Command(resume=...) 继续。
注意：必须挂 checkpointer，否则中断后状态无处保存，无法恢复。
"""

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from langchain_learn import banner, get_model, section

EXECUTED: list[str] = []


@tool
def transfer_money(to_account: str, amount: int) -> str:
    """向指定账户转账。"""
    EXECUTED.append(f"转账 {amount} 元 -> {to_account}")
    return f"已转账 {amount} 元到 {to_account}"


@tool
def get_balance(account: str) -> str:
    """查询账户余额（只读操作，无需审批）。"""
    return f"{account} 余额 8,642.00 元"


def build_agent():
    return create_agent(
        model=get_model(max_tokens=512),
        tools=[transfer_money, get_balance],
        system_prompt=(
            "你是银行 App 内的转账助手。用户已登录账户 A123，身份已核验。"
            "只要用户给出收款账户和金额，立即调用 transfer_money，"
            "不要追问转出账户，不要二次确认，不要询问用途。"
        ),
        checkpointer=InMemorySaver(),
        middleware=[
            # 只有 transfer_money 需要审批；只读工具照常执行
            HumanInTheLoopMiddleware(interrupt_on={"transfer_money": True})
        ],
    )


def invoke_until_interrupt(agent, text: str, thread_id: str):
    """发起请求，返回 (结果, config)，方便后续 resume。"""
    cfg = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke({"messages": [{"role": "user", "content": text}]}, config=cfg)
    return result, cfg


def show_interrupt(result) -> dict | None:
    """打印中断信息，返回给人工审核看的内容。"""
    interrupts = result.get("__interrupt__")
    if not interrupts:
        return None
    request = interrupts[0].value
    for action in request["action_requests"]:
        print(f"    待审批操作: {action['name']}({action['args']})")
    for cfg in request.get("review_configs", []):
        print(f"    可选决策  : {cfg['allowed_decisions']}")
    return request


def main() -> None:
    banner("1. 只读操作：不触发审批，直接执行")
    agent = build_agent()
    cfg = {"configurable": {"thread_id": "acct-demo"}}
    r = agent.invoke({"messages": [{"role": "user", "content": "查一下 A123 的余额"}]}, config=cfg)
    print(f"  回答: {r['messages'][-1].content.strip()[:60]}")
    print(f"  是否中断: {'__interrupt__' in r}")

    section("2. 转账：执行前被挂起，等人审批")
    r = agent.invoke({"messages": [{"role": "user", "content": "给 B456 转 500 元"}]}, config=cfg)
    request = show_interrupt(r)
    if request is None:
        print(f"  没触发中断，模型直接回答了：{r['messages'][-1].content[:60]}")
        return
    print(f"  此时已执行的写操作: {EXECUTED}（空的——工具还没跑）")

    section("3. 审批通过 -> 用 Command(resume) 继续")
    r = agent.invoke(Command(resume={"decisions": [{"type": "approve"}]}), config=cfg)
    print(f"  实际执行: {EXECUTED}")
    print(f"  回答: {r['messages'][-1].content.strip()[:60]}")

    section("4. 审批拒绝（带理由，模型会转述给用户）")
    r, cfg4 = invoke_until_interrupt(agent, "给 B999 转 50000 元", "acct-demo-2")
    if show_interrupt(r) is None:
        print("  没触发中断（模型选择了先追问），本段跳过")
    else:
        r = agent.invoke(
            Command(resume={"decisions": [{"type": "reject", "message": "单笔超过 1 万元需走线下审批"}]}),
            config=cfg4,
        )
        print(f"  回答: {r['messages'][-1].content.strip()[:80]}")

    section("5. 审批时修改参数（人把 500 改成 50）")
    before = len(EXECUTED)
    r, cfg5 = invoke_until_interrupt(agent, "给 B456 转 500 元", "acct-demo-3")
    if show_interrupt(r) is None:
        print("  没触发中断，本段跳过")
    else:
        r = agent.invoke(
            Command(
                resume={
                    "decisions": [
                        {
                            "type": "edit",
                            "edited_action": {
                                "name": "transfer_money",
                                "args": {"to_account": "B456", "amount": 50},
                            },
                        }
                    ]
                }
            ),
            config=cfg5,
        )
        if len(EXECUTED) > before:
            print(f"  最终执行: {EXECUTED[-1]}（模型申请的是 500，人改成了 50）")

    section("6. 四种决策")
    print("  approve  同意")
    print("  reject   拒绝（可附 message 说明原因）")
    print("  edit     改参数后执行（金额、收件人这类高危字段必备）")
    print("  respond  不执行工具，直接把一段话当作工具结果回给模型")


if __name__ == "__main__":
    main()
