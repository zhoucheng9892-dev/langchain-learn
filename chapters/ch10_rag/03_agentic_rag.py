"""第 10 章 · 示例 3：Agentic RAG —— 让模型自己决定搜不搜、搜什么

运行：
    uv run python chapters/ch10_rag/03_agentic_rag.py

固定管道（示例 2）的局限：
    无论问题是什么都检索一次，检索词就是原问题。
Agentic RAG 把检索做成工具，模型可以：
    - 判断这个问题根本不需要查资料（直接回答，省钱）；
    - 改写检索词（用户问"钩子"，资料里写"before_model"，需要转换）；
    - 多轮检索（先查 A，发现线索后再查 B）。

代价：多几次模型调用，延迟和成本都上升。要不要上，取决于问题复杂度。
"""

from langchain.agents import create_agent
from langchain.tools import tool

from langchain_learn import banner, get_model, section
from langchain_learn.rag import build_retriever

retriever = build_retriever(k=3)


@tool
def search_docs(query: str) -> str:
    """在内部技术文档中检索资料。query 用关键词，不要用整句疑问句。"""
    docs = retriever.invoke(query)
    if not docs:
        return "未检索到相关资料。"
    blocks = []
    for i, d in enumerate(docs, 1):
        blocks.append(f"[{i}] 来源={d.metadata['source']} 小节={d.metadata['headers']}\n{d.page_content}")
    return "\n\n".join(blocks)


@tool
def word_count(text: str) -> int:
    """统计一段文本的字数（不需要联网的本地工具）。"""
    return len(text)


def main() -> None:
    agent = create_agent(
        model=get_model(max_tokens=768),
        tools=[search_docs, word_count],
        system_prompt=(
            "你是技术文档助手。\n"
            "规则：\n"
            "1. 涉及 LangChain 具体事实（版本、参数、清单）时必须先调用 search_docs；\n"
            "2. 常识性问题（打招呼、简单计算）直接回答，不要检索；\n"
            "3. 检索词要提炼成关键词，不要照抄整句；\n"
            "4. 回答末尾标注来源文件名。"
        ),
    )

    banner("1. 需要查资料的问题：会触发检索")
    r = agent.invoke(
        {"messages": [{"role": "user", "content": "SummarizationMiddleware 的 trigger 支持哪几种阈值？"}]}
    )
    for m in r["messages"]:
        if getattr(m, "tool_calls", None):
            print(f"  -> 调用 {[c['name'] + '(' + str(c['args']) + ')' for c in m.tool_calls]}")
    print(f"  回答: {str(r['messages'][-1].content).strip()[:200]}")

    section("2. 不需要查资料的问题：直接回答，零检索")
    r2 = agent.invoke({"messages": [{"role": "user", "content": "你好，随便聊两句就行。"}]})
    called = any(getattr(m, "tool_calls", None) for m in r2["messages"])
    print(f"  是否检索: {called}")
    print(f"  回答: {str(r2['messages'][-1].content).strip()[:80]}")

    section("3. 需要多步推理：先查清单，再查细节")
    r3 = agent.invoke(
        {"messages": [{"role": "user", "content": "内置中间件里，哪个能防止 Agent 停不下来？它怎么配？"}]}
    )
    for m in r3["messages"]:
        if getattr(m, "tool_calls", None):
            print(f"  -> 第 {len([x for x in r3['messages'] if getattr(x,'tool_calls',None)])} 次调用: "
                  f"{[c['args'] for c in m.tool_calls]}")
    print(f"  回答: {str(r3['messages'][-1].content).strip()[:220]}")

    section("4. 固定管道 vs Agentic RAG")
    print("  固定管道    : 1 次检索 + 1 次生成，稳定、便宜、可预测")
    print("  Agentic RAG : N 次检索 + M 次生成，灵活、能多跳，但成本延迟上升")
    print("  建议：先做固定管道，只有确实需要多跳/条件检索时再升级。")


if __name__ == "__main__":
    main()
