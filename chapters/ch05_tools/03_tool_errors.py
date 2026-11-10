"""第 5 章 · 示例 3：工具出错了怎么办（重试 / 降级 / 转述）

运行：
    uv run python chapters/ch05_tools/03_tool_errors.py

工具的失败分三种，处理方式完全不同：
    1. 业务性失败（查不到、无权限）—— 别抛异常，return 一句人话，模型会转述；
    2. 偶发性失败（网络抖动、限流）—— 用 ToolRetryMiddleware 自动重试；
    3. 代码性失败（KeyError、除零）—— 用 ToolErrorMiddleware 统一兜底，避免整轮崩掉。
"""

from langchain.agents import create_agent
from langchain.agents.middleware import ToolErrorMiddleware, ToolRetryMiddleware
from langchain.tools import tool

from langchain_learn import banner, get_model, section

CALLS = {"flaky": 0}  # 记录调用次数，模拟"前两次失败第三次成功"


@tool
def flaky_api(city: str) -> str:
    """查询城市空气质量（模拟不稳定的第三方接口）。"""
    CALLS["flaky"] += 1
    if CALLS["flaky"] < 3:
        raise ConnectionError(f"第 {CALLS['flaky']} 次请求超时")
    return f"{city} AQI=42（优）"


@tool
def divide(a: int, b: int) -> int:
    """做除法 a / b。"""
    return a // b  # b=0 时会抛 ZeroDivisionError


@tool
def business_lookup(order_id: str) -> str:
    """按订单号查询订单状态。"""
    # 业务失败：不抛异常，返回可读信息
    return f"未找到订单 {order_id}，请确认订单号是否正确。"


def run(agent, question: str) -> str:
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    return str(result["messages"][-1].content).strip()


def main() -> None:
    banner("1. 业务性失败：return 一句人话，模型自己会解释")

    agent = create_agent(
        model=get_model(max_tokens=512),
        tools=[business_lookup],
        system_prompt="你是客服助手。查不到时要如实告知，不要编造。",
    )
    print(f"  {run(agent, '帮我查订单 A999 的状态')}")

    section("2. 偶发失败：ToolRetryMiddleware 自动重试")
    retry_agent = create_agent(
        model=get_model(max_tokens=512),
        tools=[flaky_api],
        system_prompt="你是环境助手。",
        # 指数退避，最多重试 3 次；只在异常时重试
        middleware=[ToolRetryMiddleware(max_retries=3, initial_delay=0.2, backoff_factor=2.0)],
    )
    answer = run(retry_agent, "查一下成都的空气质量")
    print(f"  答案: {answer}")
    print(f"  工具实际被调用: {CALLS['flaky']} 次（前两次抛异常，被中间件吃掉并重试）")

    section("3. 代码性失败：ToolErrorMiddleware 统一兜底")

    def on_error(exc: Exception, request) -> str:
        """把任何异常翻译成模型能理解的一句话。"""
        tool_name = request.tool_call["name"] if hasattr(request, "tool_call") else "unknown"
        print(f"    [兜底] {tool_name} 抛出 {type(exc).__name__}: {exc}")
        return f"工具 {tool_name} 执行失败（{type(exc).__name__}）。请换一种方式，或告知用户暂时无法完成。"

    error_agent = create_agent(
        model=get_model(max_tokens=512),
        tools=[divide],
        system_prompt="你是计算助手。",
        middleware=[ToolErrorMiddleware(on_error)],
    )
    print(f"  {run(error_agent, '帮我算一下 10 除以 0 等于多少')}")

    section("4. 不装中间件会怎样？")
    plain_agent = create_agent(model=get_model(max_tokens=512), tools=[divide], system_prompt="你是计算助手。")
    try:
        print(f"  {run(plain_agent, '算一下 10 除以 0')}")
    except Exception as exc:  # noqa: BLE001
        print(f"  整轮调用直接失败: {type(exc).__name__}: {str(exc)[:100]}")
        print("  这就是为什么生产环境一定要装 ToolErrorMiddleware。")

    section("5. 决策表")
    print("  查不到 / 无权限 / 参数不合理  -> return 提示语")
    print("  网络、限流、超时              -> ToolRetryMiddleware")
    print("  代码 bug、外部服务契约变更    -> ToolErrorMiddleware + 告警")


if __name__ == "__main__":
    main()
