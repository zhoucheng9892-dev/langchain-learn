"""第 7 章 · 示例 1：自定义中间件 —— v1 最重要的扩展点

运行：
    uv run python chapters/ch07_middleware/01_custom_middleware.py

Agent 的一次运行有 6 个可插入的时机：
    before_agent -> before_model -> (模型) -> after_model -> (工具) -> after_agent
                                     wrap_model_call / wrap_tool_call 可以包住整段调用

用中间件能做什么：改提示词、注入上下文、限制工具、审计日志、脱敏、兜底重试……
原则：能用中间件解决的，不要写进提示词（提示词不可靠，代码才可靠）。
"""

import time
from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    ModelRequest,
    after_model,
    before_model,
    dynamic_prompt,
    wrap_tool_call,
)
from langchain.messages import AIMessage, SystemMessage
from langchain.tools import tool

from langchain_learn import banner, get_model, section


@tool
def get_balance(account: str) -> str:
    """查询账户余额。"""
    return f"{account} 余额 8,642.00 元"


@dataclass
class BankContext:
    """每次调用传入的上下文（用户身份、套餐等）。"""

    plan: str = "free"


# ---- 1. 动态提示词：每次调用前按运行时信息生成 system prompt --------------
@dynamic_prompt
def role_prompt(request) -> SystemMessage:
    """根据 context 决定人设，比写死 system_prompt 灵活得多。"""
    plan = getattr(request.runtime.context, "plan", "free")
    tone = "你是资深理财顾问，回答专业、给出行动建议。" if plan == "pro" else "你是基础客服，回答简洁，不要做投资建议。"
    return SystemMessage(tone)


# ---- 2. before_model / after_model：观察 + 改写 -------------------------
class AuditMiddleware(AgentMiddleware):
    """记录每一步耗时与 token，并统计工具调用次数。"""

    def before_model(self, state, runtime):
        self._t0 = time.perf_counter()
        print(f"    [audit] 进入模型前，消息数={len(state['messages'])}")
        return None  # 返回 dict 可以更新 state；返回 None 表示不改

    def after_model(self, state, runtime):
        elapsed = time.perf_counter() - self._t0
        last = state["messages"][-1]
        tokens = (last.usage_metadata or {}).get("total_tokens", 0) if isinstance(last, AIMessage) else 0
        print(f"    [audit] 模型返回，耗时 {elapsed * 1000:.0f}ms，token {tokens}")
        return None


# ---- 3. wrap_tool_call：在工具外面套一层（最常用） ----------------------
@wrap_tool_call
def tool_guard(request, handler):
    """三道保险：参数脱敏打印、敏感操作拦截、结果截断。"""
    name = request.tool_call["name"]
    args = request.tool_call["args"]
    print(f"    [guard] 准备调用 {name}({args})")

    if name == "transfer_money":
        # 硬拦截：不管模型多想转，代码说不行就是不行
        from langchain.messages import ToolMessage

        return ToolMessage(
            content="拒绝：转账功能需要人工审批，请引导用户走 App 内流程。",
            tool_call_id=request.tool_call["id"],
            name=name,
        )

    result = handler(request)  # 真正执行工具
    print(f"    [guard] {name} 返回 {str(result.content)[:40]}")
    return result


# ---- 4. before_model 装饰器写法：注入额外上下文 -------------------------
@before_model
def inject_context(state, runtime):
    """在模型看到消息前，塞一条"系统提醒"进去（示例：当前时间）。"""
    if any(isinstance(m, SystemMessage) and "当前时间" in str(m.content) for m in state["messages"]):
        return None
    return {
        "messages": state["messages"]
        + [SystemMessage(f"当前时间：{time.strftime('%Y-%m-%d %H:%M')}，涉及时间问题以此为准。")]
    }


@tool
def transfer_money(to_account: str, amount: int) -> str:
    """向指定账户转账。"""
    return f"已转账 {amount} 元到 {to_account}"


def main() -> None:
    banner("1. 组装带中间件的 Agent")

    agent = create_agent(
        model=get_model(max_tokens=512),
        tools=[get_balance, transfer_money],
        system_prompt="你是银行助手。",  # 会被 dynamic_prompt 覆盖
        context_schema=BankContext,
        middleware=[
            role_prompt,          # 动态人设
            AuditMiddleware(),    # 可观测
            tool_guard,           # 工具护栏
            inject_context,       # 上下文注入
        ],
    )

    section("2. free 用户问余额")
    r1 = agent.invoke(
        {"messages": [{"role": "user", "content": "我的账户 A123 还有多少钱？"}]},
        context=BankContext(plan="free"),
    )
    print(f"  回答: {r1['messages'][-1].content.strip()[:80]}")

    section("3. 试图转账：被 wrap_tool_call 硬拦截")
    r2 = agent.invoke(
        {"messages": [{"role": "user", "content": "从 A123 转 500 元到 B456"}]},
        context=BankContext(plan="pro"),
    )
    print(f"  回答: {r2['messages'][-1].content.strip()[:120]}")

    section("4. pro 用户问投资建议（人设已切换）")
    r3 = agent.invoke(
        {"messages": [{"role": "user", "content": "我有 5 万闲钱，怎么配置比较好？"}]},
        context=BankContext(plan="pro"),
    )
    print(f"  回答: {r3['messages'][-1].content.strip()[:100]}")

    section("5. 小结")
    print("  dynamic_prompt  : 按 context / state 动态生成 system prompt")
    print("  before_model    : 改 state（注入上下文、裁剪历史）、打日志")
    print("  after_model     : 校验输出、统计 token、触发告警")
    print("  wrap_tool_call  : 工具级权限、脱敏、重试、结果改写")
    print("  after_agent     : 整轮结束后的归档、评测采样")


if __name__ == "__main__":
    main()
