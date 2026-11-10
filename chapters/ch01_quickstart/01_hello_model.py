"""第 1 章 · 示例 1：最小可运行的模型调用

运行：
    uv run python chapters/ch01_quickstart/01_hello_model.py

本章要建立的第一个认知：
    在 LangChain 1.x 里，"调用大模型"这件事不再需要 Chain、Memory、AgentExecutor，
    一个 ChatModel 实例就够了。Agent 只是在它外面加了一个"循环 + 工具"。
"""

from langchain_learn import banner, get_model, section


def main() -> None:
    # ---- 1. 拿到模型实例 -------------------------------------------------
    # get_model() 内部是 ChatDeepSeek(model="deepseek-flash", reasoning_effort="none")
    # 关闭思考模式是为了让示例更快、更省钱；需要推理能力时传 thinking=True。
    model = get_model()

    banner("1. invoke：一次性拿到完整结果")
    reply = model.invoke("用两句话说明 LangChain 1.x 是什么。")
    print(reply.content)

    # 返回值是一条 AIMessage，不是字符串。字符串藏在 .content 里，
    # 元信息（token 用量、结束原因、模型名）都在 .usage_metadata / .response_metadata
    section("2. 返回值是 AIMessage，不是 str")
    print(f"类型      : {type(reply).__name__}")
    print(f"内容长度  : {len(reply.content)} 字")
    print(f"token 用量: {reply.usage_metadata}")
    print(f"结束原因  : {reply.response_metadata.get('finish_reason')}")

    # ---- 2. 流式输出 -----------------------------------------------------
    # 打字机效果靠的就是 stream()。注意：此时 content 是"增量片段"，不是完整答案。
    section("3. stream：逐字返回，适合做打字机效果")
    for chunk in model.stream("列出 3 个学习 LangChain 的建议，每个不超过 15 字。"):
        print(chunk.content, end="", flush=True)
    print()

    # ---- 3. 批量调用 -----------------------------------------------------
    # batch() 会自动并发，比 for 循环快很多，且只暴露一次 SDK 调用逻辑。
    section("4. batch：并发处理多条输入")
    questions = ["一句话解释什么是 RAG", "一句话解释什么是 Agent"]
    answers = model.batch(questions)
    for q, a in zip(questions, answers):
        print(f"Q: {q}\nA: {a.content.strip()}\n")


if __name__ == "__main__":
    main()
