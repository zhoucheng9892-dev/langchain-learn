"""第 8 章 · 示例 3：历史太长怎么办 —— SummarizationMiddleware

运行：
    uv run python chapters/ch08_memory/03_summarization.py

trim_messages 的做法是"丢掉老消息"，代价是忘事。
SummarizationMiddleware 的做法是"把老消息压成一段摘要"，既省 token 又保住关键信息。
它在 before_model 里检查触发条件，一旦超过阈值就把前面的消息替换成一条摘要消息。
"""

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from langchain_learn import banner, get_model, section

# 一段有信息量的多轮对话：越往后，关键约束越容易被冲掉
DIALOG = [
    "我要去成都出差三天，帮我记一下：预算是 3000 元以内。",
    "我偏好住春熙路附近，方便吃夜宵。",
    "我不吃辣，肠胃不太好。",
    "11 月 5 日出发，早班机最好。",
    "顺便说一下，我需要带一个投影仪做演示。",
    "现在请给我一个初步的行程安排。",
]


def run(with_summary: bool) -> list:
    middleware = []
    if with_summary:
        middleware.append(
            SummarizationMiddleware(
                model=get_model(max_tokens=512),
                trigger=("messages", 8),   # 消息数超过 8 条就触发
                keep=("messages", 2),      # 保留最近 2 条原文
            )
        )
    agent = create_agent(
        model=get_model(max_tokens=512),
        tools=[],
        system_prompt="你是差旅助手。基于已知信息给出安排，缺信息就问。",
        checkpointer=InMemorySaver(),
        middleware=middleware,
    )
    cfg = {"configurable": {"thread_id": f"trip-{'sum' if with_summary else 'raw'}"}}
    for i, text in enumerate(DIALOG, 1):
        r = agent.invoke({"messages": [HumanMessage(text)]}, config=cfg)
    state = agent.get_state(cfg)
    return state.values["messages"]


def main() -> None:
    banner("1. 不压缩：消息一直累积")
    raw = run(with_summary=False)
    print(f"  最终消息数: {len(raw)}")
    print(f"  最后一条回答: {str(raw[-1].content).strip()[:70]}")

    section("2. 开启摘要中间件：历史被压成摘要")
    summed = run(with_summary=True)
    print(f"  最终消息数: {len(summed)}  （明显变少）")
    print()
    print("  压缩后的消息序列：")
    for m in summed:
        kind = type(m).__name__
        text = str(m.content).replace("\n", " ")[:90]
        print(f"    {kind:<14} {text}")

    section("3. 关键：摘要里保住了哪些约束？")
    joined = " ".join(str(m.content) for m in summed)
    for keyword in ["3000", "春熙路", "不吃辣", "投影仪", "11 月 5 日"]:
        print(f"    {'✓' if keyword in joined else '✗'} {keyword}")

    section("4. 参数怎么调")
    print("  trigger=('messages', 8)   消息数阈值")
    print("  trigger=('tokens', 4000)  按 token 阈值（更贴近真实成本）")
    print("  trigger=('fraction', 0.8) 按上下文窗口占比")
    print("  keep=('messages', 20)     永远保留最近 N 条原文")
    print("  注意：摘要本身也要花一次模型调用，阈值别调太低。")


if __name__ == "__main__":
    main()
