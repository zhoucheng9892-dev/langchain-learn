"""第 3 章 · 示例 3：上下文裁剪 trim_messages

运行：
    uv run python chapters/ch03_messages/03_trim_messages.py

Agent 跑久了 messages 会越来越长，最终超出上下文窗口、或者账单暴涨。
trim_messages 是"最后一道保险"：保留 system + 最近若干条，把老的丢掉。
第 7 章的 SummarizationMiddleware 是"更聪明的那一道"：老的压缩成摘要而不是丢掉。
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain.messages import trim_messages
from langchain_core.messages.utils import count_tokens_approximately

from langchain_learn import banner, get_model, section


def build_history(turns: int = 12):
    """造一段长对话：每轮一个问 + 一个答。"""
    history = [SystemMessage("你是一个购物助手，回答简短。")]
    for i in range(1, turns + 1):
        history.append(HumanMessage(f"第 {i} 个问题：这件商品有货吗？"))
        history.append(AIMessage(f"第 {i} 个回答：有货，预计 2 天送达。"))
    return history


def main() -> None:
    model = get_model(max_tokens=128)
    history = build_history()
    history.append(HumanMessage("刚才我问了几次有没有货？"))

    banner("1. 不裁剪：把整段历史都发过去")
    total = count_tokens_approximately(history)
    print(f"  消息数: {len(history)}   估算 token: {total}")

    section("2. trim_messages：只保留最近的消息，且不超过 token 预算")
    trimmed = trim_messages(
        history,
        max_tokens=120,            # 预算
        strategy="last",           # 从最新往回保留
        token_counter=count_tokens_approximately,
        include_system=True,       # system 永远保留
        allow_partial=False,       # 不切半条消息
        start_on="human",          # 保证裁剪后第一条是 human，避免结构错乱
    )
    print(f"  裁剪后消息数: {len(trimmed)}   估算 token: {count_tokens_approximately(trimmed)}")
    print(f"  第一条类型: {type(trimmed[0]).__name__}")
    print(f"  最后一条: {trimmed[-1].content}")

    section("3. 把裁剪放进管道，每次调用自动生效")
    trimmed_answer = model.invoke(trimmed)
    print(f"  模型回答: {trimmed_answer.content.strip()}")
    print("  注意：模型看不到第 1-5 轮，所以它的答案可能不完整——")
    print("  这就是'丢历史'的代价。要既省 token 又不丢信息，请用摘要（第 7 章）。")


if __name__ == "__main__":
    main()
