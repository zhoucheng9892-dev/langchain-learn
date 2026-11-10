"""第 12 章 · 示例 1：可观测性 —— 自己统计 token、耗时与工具调用

运行：
    uv run python chapters/ch12_production/01_observability.py

上线之后你一定会问三个问题：
    这次请求花了多少 token？慢在哪一步？工具调用了几次？
CallbackHandler 是本地答案；跨会话排查用 LangSmith。
两者不冲突：本地 handler 负责实时打点与告警，LangSmith 负责事后追溯。
"""

import time
from collections import defaultdict

from langchain_core.callbacks import BaseCallbackHandler

from langchain.agents import create_agent
from langchain.tools import tool

from langchain_learn import MODEL_REF, banner, get_model, section


class MetricsHandler(BaseCallbackHandler):
    """统计：模型调用次数/token/耗时，工具调用次数/耗时。"""

    def __init__(self) -> None:
        self.llm_calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.tool_calls = defaultdict(int)
        self.tool_ms = defaultdict(float)
        self._t0: float = 0.0
        self.total_ms = 0.0

    # ---- 模型 -----------------------------------------------------------
    def on_chat_model_start(self, serialized, messages, **kwargs):
        self.llm_calls += 1
        self._t0 = time.perf_counter()

    def on_llm_end(self, response, **kwargs):
        self.total_ms += (time.perf_counter() - self._t0) * 1000
        usage = getattr(response, "llm_output", None) or {}
        tokens = usage.get("token_usage") or {}
        self.prompt_tokens += tokens.get("prompt_tokens", 0)
        self.completion_tokens += tokens.get("completion_tokens", 0)

    # ---- 工具 -----------------------------------------------------------
    def on_tool_start(self, serialized, input_str, **kwargs):
        self._t0 = time.perf_counter()

    def on_tool_end(self, output, **kwargs):
        name = kwargs.get("name") or "unknown"
        self.tool_calls[name] += 1
        self.tool_ms[name] += (time.perf_counter() - self._t0) * 1000

    # ---- 汇总 -----------------------------------------------------------
    def report(self) -> str:
        cost_in = self.prompt_tokens / 1_000_000 * 0.5   # 示例单价，按实际报价替换
        cost_out = self.completion_tokens / 1_000_000 * 2.0
        lines = [
            f"模型调用 {self.llm_calls} 次，累计 {self.total_ms:.0f}ms",
            f"token: 输入 {self.prompt_tokens} / 输出 {self.completion_tokens}",
            f"估算成本: ¥{cost_in + cost_out:.6f}",
        ]
        for name, count in self.tool_calls.items():
            lines.append(f"工具 {name}: {count} 次，累计 {self.tool_ms[name]:.0f}ms")
        return "\n  ".join(lines)


@tool
def search_order(order_id: str) -> str:
    """按订单号查询订单。"""
    time.sleep(0.05)  # 模拟一次网络请求
    return f"订单 {order_id}: 已发货，预计 11 月 2 日送达"


@tool
def refund(order_id: str, reason: str) -> str:
    """为订单发起退款。"""
    time.sleep(0.05)
    return f"订单 {order_id} 退款已受理，原因：{reason}"


def main() -> None:
    banner("1. 挂上 handler，跑一次完整请求")
    handler = MetricsHandler()

    agent = create_agent(
        model=get_model(max_tokens=512),
        tools=[search_order, refund],
        system_prompt="你是售后助手，先查订单再决定是否退款。回答简短。",
    )

    r = agent.invoke(
        {"messages": [{"role": "user", "content": "订单 A100 一直没到，帮我看看，然后申请退款，原因写「超时未送达」。"}]},
        config={"callbacks": [handler], "run_name": "refund-flow"},
    )
    print(f"  回答: {str(r['messages'][-1].content).strip()[:80]}")

    section("2. 本次请求的指标")
    print(f"  {handler.report()}")

    section("3. 用 usage_metadata 交叉验证（AIMessage 自带）")
    total = {"input": 0, "output": 0}
    for m in r["messages"]:
        usage = getattr(m, "usage_metadata", None)
        if usage:
            total["input"] += usage["input_tokens"]
            total["output"] += usage["output_tokens"]
    print(f"  消息累计 token: 输入 {total['input']} / 输出 {total['output']}")
    print("  说明：callback 拿到的是 SDK 层用量，usage_metadata 是消息层用量，两者应接近。")

    section("4. 接 LangSmith：改环境变量即可，不用改代码")
    print("  LANGSMITH_TRACING=true")
    print("  LANGSMITH_API_KEY=lsv2_xxx")
    print("  LANGSMITH_PROJECT=langchain-deepseek-learning")
    print(f"  也可以在代码里给 config 加 tags/metadata，便于筛选：")
    print(f"    model ref = {MODEL_REF}")
    print('    config={"tags": ["prod", "refund"], "metadata": {"user_id": "u_1"}}')

    section("5. 该埋哪些点")
    print("  - 每次请求: thread_id / user_id / 模型名 / 总 token / 总耗时")
    print("  - 每次工具: 工具名 / 参数摘要 / 耗时 / 是否失败")
    print("  - 异常: 捕获后带上 run_id，方便和 LangSmith 对齐")


if __name__ == "__main__":
    main()
