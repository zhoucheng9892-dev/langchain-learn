"""第 10 章 · 示例 2：完整 RAG 问答（含"检索 vs 不检索"对比）

运行：
    uv run python chapters/ch10_rag/02_rag_chain.py

这段 40 行的管道就是所谓 RAG：
    检索 -> 拼进提示词 -> 生成 -> 附上来源
关键在提示词里要明确"只依据资料回答，资料没有就说不知道"，
否则模型会用自己的参数记忆去补，反而比不检索更危险。
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langchain_learn import banner, get_model, section
from langchain_learn.rag import build_retriever

# 系统提示词是整个 RAG 的"缰绳"
RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一个严谨的技术文档助手。规则：\n"
            "1. 只能依据 <context> 中的资料回答；\n"
            "2. 资料里没有的内容，直接说「资料中未提及」，不要编造；\n"
            "3. 回答末尾用「来源：文件名」标注依据；\n"
            "4. 回答控制在 150 字以内。",
        ),
        ("human", "<context>\n{context}\n</context>\n\n问题：{question}"),
    ]
)


def format_docs(docs) -> str:
    """把检索结果拼成带来源标记的上下文。"""
    blocks = []
    for i, doc in enumerate(docs, 1):
        blocks.append(f"[片段{i} | {doc.metadata['source']} | {doc.metadata['headers']}]\n{doc.page_content}")
    return "\n\n".join(blocks)


def main() -> None:
    model = get_model(max_tokens=512)
    retriever = build_retriever(k=3)

    questions = [
        "1.x 里 create_agent 取代了哪个函数？",
        "生产环境 recursion_limit 一般设多少？超了会怎样？",
        "LangChain 默认支持哪些向量数据库？（这条资料里其实没有）",
    ]

    banner("1. 不检索：直接问模型（容易张冠李戴）")
    for q in questions:
        answer = model.invoke(q).content.strip().replace("\n", " ")
        print(f"  Q: {q}\n  A: {answer[:90]}\n")

    section("2. 有检索：先找资料，再作答")
    chain = RAG_PROMPT | model | StrOutputParser()
    for q in questions:
        docs = retriever.invoke(q)
        answer = chain.invoke({"context": format_docs(docs), "question": q})
        print(f"  Q: {q}")
        print(f"  检索到: {[(d.metadata['source'], d.metadata['headers']) for d in docs]}")
        print(f"  A: {answer.strip()}\n")

    section("3. 拆开看检索环节")
    q = questions[0]
    for i, doc in enumerate(retriever.invoke(q), 1):
        print(f"  [{i}] {doc.metadata['source']} / {doc.metadata['headers']}")
        print(f"      {doc.page_content[:80].replace(chr(10), ' ')}...")

    section("4. 工程要点")
    print("  - 检索为空时不要让模型硬答，直接返回「未找到相关资料」；")
    print("  - 上下文要带来源，方便前端展示引用、也方便排查幻觉；")
    print("  - 提示词里写死「资料没有就说不知道」，比事后审核便宜得多。")


if __name__ == "__main__":
    main()
