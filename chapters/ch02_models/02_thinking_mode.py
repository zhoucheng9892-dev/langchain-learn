"""第 2 章 · 示例 2：思考模式（reasoning）怎么用、会踩什么坑

运行：
    uv run python chapters/ch02_models/02_thinking_mode.py

DeepSeek v4 系列默认开启思考：模型先输出一段 reasoning_content，再输出答案。
LangChain 不会把思考内容混进 .content，而是放在 additional_kwargs["reasoning_content"]。
这带来三个直接影响：
    1. 输出 token 变多（思考也算钱），响应变慢；
    2. max_tokens 太小时，额度被思考吃光，content 会是空字符串；
    3. 思考模式下不允许强制 tool_choice，结构化输出会 400。
"""

from langchain_learn import banner, get_model, section


def ask_thinking(question: str):
    """开启思考模式问一次，返回 (答案, 思考内容, usage)。"""
    model = get_model(thinking=True, max_tokens=2048)
    msg = model.invoke(question)
    reasoning = (msg.additional_kwargs or {}).get("reasoning_content", "")
    return msg.content, reasoning, msg.usage_metadata


def main() -> None:
    question = "一个水池有两个进水管和一个出水管，甲管 4 小时注满，乙管 6 小时注满，丙管 12 小时排空。三管同开多久注满？"

    banner("1. 关闭思考 vs 开启思考")
    fast = get_model(thinking=False, max_tokens=512)
    msg_fast = fast.invoke(question)
    print("【关闭思考】")
    print(f"  答案     : {msg_fast.content.strip()}")
    print(f"  输出token: {msg_fast.usage_metadata['output_tokens']}")
    print(f"  思考内容 : 无")

    content, reasoning, usage = ask_thinking(question)
    print("\n【开启思考】")
    print(f"  答案     : {content.strip()[:120]}")
    print(f"  输出token: {usage['output_tokens']}（其中思考 {usage.get('output_token_details', {}).get('reasoning', 0)}）")
    print(f"  思考内容 : {reasoning.strip()[:160].replace(chr(10), ' ')}...")

    section("2. 流式读取思考过程（做「思考中...」的 UI 就靠它）")
    model = get_model(thinking=True, max_tokens=1024)
    in_thought = True
    for chunk in model.stream("简述快速排序的核心思想，一句话。"):
        delta = (chunk.additional_kwargs or {}).get("reasoning_content")
        if delta:
            print(delta, end="", flush=True)
        elif chunk.content:
            if in_thought:
                print("\n--- 思考结束，开始输出答案 ---")
                in_thought = False
            print(chunk.content, end="", flush=True)
    print()

    section("3. 坑：max_tokens 太小，答案被思考挤没")
    tiny = get_model(thinking=True, max_tokens=20)
    msg = tiny.invoke("9.11 和 9.9 哪个大？为什么？")
    print(f"  content     = {msg.content!r}")
    print(f"  finish_reason = {msg.response_metadata.get('finish_reason')}")
    print(f"  usage       = {msg.usage_metadata}")
    print("  说明：思考把 20 个 token 用完了，正文一个字都没轮到。")
    print("  结论：开思考时 max_tokens 至少给 1024，或直接用 reasoning_effort='none'。")


if __name__ == "__main__":
    main()
