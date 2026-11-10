"""第 12 章 · 示例 2：最小可用的评测（回归测试）

运行：
    uv run python chapters/ch12_production/02_eval.py

"感觉效果变好了"不是工程结论。每次改提示词、换模型、调检索参数，都要跑一遍回归。
评测三步：
    1. 数据集：20~50 条真实问题，带标准答案或关键词；
    2. 打分器：能关键词匹配就别上模型（便宜、稳定）；
    3. 报告：通过率 + 失败样本，失败样本比总分更有用。
"""

import re

from langchain_core.output_parsers import StrOutputParser
from langchain_learn import banner, get_model, section
from langchain_learn.rag import build_retriever
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# ---- 1. 数据集：问题 + 必须出现的关键词 ---------------------------------
DATASET = [
    {
        "q": "create_agent 取代了哪个旧函数？",
        "must": ["create_react_agent"],
    },
    {
        "q": "1.x 里 Chain 相关的旧代码放到哪个包了？",
        "must": ["langchain-classic"],
    },
    {
        "q": "SummarizationMiddleware 的 trigger 支持哪几种阈值？",
        "must": ["messages", "tokens", "fraction"],
    },
    {
        "q": "生产环境 recursion_limit 一般设多少？",
        "must": ["10", "30"],
    },
    {
        "q": "人工审批中间件用 Command 恢复时，decisions 里可以填哪几种 type？",
        "must": ["approve"],
    },
]

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "只依据 <context> 回答，资料里没有就说「资料中未提及」。回答精炼，保留专有名词原文。",
        ),
        ("human", "<context>\n{context}\n</context>\n\n问题：{question}"),
    ]
)


def build_chain():
    model = get_model(max_tokens=384)
    return PROMPT | model | StrOutputParser()


def score_by_keywords(answer: str, must: list[str]) -> tuple[bool, list[str]]:
    """关键词打分：全部命中才算通过（适合有明确术语的场景）。"""
    missing = [k for k in must if k.lower() not in answer.lower()]
    return (not missing, missing)


# ---- 2. LLM-as-judge：主观题才用，且必须给评分标准 -----------------------
class JudgeResult(BaseModel):
    score: int = Field(ge=1, le=5, description="1-5 分")
    reason: str = Field(description="一句话理由")


JUDGE_PROMPT = """你是严格的评审。根据下面的标准给回答打分。

评分标准：
5 分：完全正确，术语准确，无多余内容
4 分：正确但略有冗余
3 分：方向对，但关键细节缺失或有小错
2 分：大部错误
1 分：完全错误或答非所问

问题：{question}
标准答案要点：{expected}
待评回答：{answer}
"""


def judge(question: str, expected: str, answer: str) -> JudgeResult:
    judge_model = get_model(max_tokens=256).with_structured_output(JudgeResult)
    return judge_model.invoke(JUDGE_PROMPT.format(question=question, expected=expected, answer=answer))


def main() -> None:
    retriever = build_retriever(k=3)
    chain = build_chain()

    banner("1. 跑一遍数据集（关键词判定）")
    passed = 0
    failures = []
    for i, case in enumerate(DATASET, 1):
        docs = retriever.invoke(case["q"])
        context = "\n\n".join(f"[{d.metadata['source']}] {d.page_content}" for d in docs)
        answer = chain.invoke({"context": context, "question": case["q"]}).strip()
        ok, missing = score_by_keywords(answer, case["must"])
        passed += ok
        flag = "PASS" if ok else "FAIL"
        print(f"  {i}. [{flag}] {case['q']}")
        print(f"       答案: {answer[:70].replace(chr(10), ' ')}")
        if not ok:
            print(f"       缺少关键词: {missing}")
            failures.append({"q": case["q"], "answer": answer, "missing": missing})

    section("2. 结果")
    rate = passed / len(DATASET) * 100
    print(f"  通过率 {passed}/{len(DATASET)} = {rate:.0f}%")
    if failures:
        print("  失败样本（比总分更有价值）：")
        for f in failures:
            print(f"    - {f['q']} -> 缺 {f['missing']}")

    section("3. LLM-as-judge：给主观题打分")
    demo_q = DATASET[2]["q"]
    docs = retriever.invoke(demo_q)
    context = "\n\n".join(d.page_content for d in docs)
    answer = chain.invoke({"context": context, "question": demo_q}).strip()
    verdict = judge(demo_q, "messages / tokens / fraction 三种阈值", answer)
    print(f"  问题: {demo_q}")
    print(f"  回答: {answer[:80]}")
    print(f"  评分: {verdict.score}/5  理由: {verdict.reason}")

    section("4. 评测工程建议")
    print("  - 数据集从真实用户问题里抽样，别自己编；")
    print("  - 能规则判定就别用模型判定（模型判定也有方差，建议跑 3 次取均值）；")
    print("  - 每次改动都记录：改动内容 / 通过率 / 平均 token / 平均延迟；")
    print("  - 通过率下降 2 个点就回滚，不要靠感觉赌。")


if __name__ == "__main__":
    main()
