"""第 3 章 · 示例 2：Prompt 模板与 LCEL 管道

运行：
    uv run python chapters/ch03_messages/02_prompt_template.py

v1 里 Chain 基本消失了，但"模板 + 管道"依然是最常用的组合：
    prompt | model | parser
| 是 LCEL 的语法糖，本质是 RunnableSequence，天生支持 invoke/stream/batch/ainvoke。
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.messages import HumanMessage

from langchain_learn import banner, get_model, section


def main() -> None:
    model = get_model(max_tokens=256)

    banner("1. 最基础的模板：变量用 {} 占位")
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是{company}的客服，语气专业、礼貌，回答不超过 40 字。"),
            ("human", "{question}"),
        ]
    )
    # 模板本身也是 Runnable：可以先单独格式化，看最终发给模型的是什么
    formatted = prompt.invoke({"company": "星河云", "question": "你们支持退款吗？"})
    print(f"  格式化后的消息: {[(type(m).__name__, m.content) for m in formatted.messages]}")

    section("2. 组装管道：prompt | model | parser")
    chain = prompt | model | StrOutputParser()
    answer = chain.invoke({"company": "星河云", "question": "你们支持退款吗？"})
    print(f"  回答: {answer.strip()}")

    section("3. 链也可以流式和批量")
    print("  流式: ", end="")
    for token in chain.stream({"company": "星河云", "question": "说一句欢迎语。"}):
        print(token, end="", flush=True)
    print()

    answers = chain.batch(
        [
            {"company": "星河云", "question": "发票怎么开？"},
            {"company": "星河云", "question": "支持哪些付款方式？"},
        ]
    )
    print(f"  批量: 共 {len(answers)} 条回答")

    section("4. MessagesPlaceholder：把历史对话塞进模板")
    chat_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一个记忆助手。"),
            MessagesPlaceholder("history"),  # 历史消息列表插入点
            ("human", "{question}"),
        ]
    )
    filled = chat_prompt.invoke(
        {
            "history": [HumanMessage("我最喜欢的颜色是靛蓝。"), HumanMessage("记住它。")],
            "question": "我最喜欢什么颜色？只答颜色。",
        }
    )
    print(f"  模板产出了 {len(filled.messages)} 条消息")
    print(f"  模型回答: {model.invoke(filled).content.strip()}")

    section("5. partial：预设一部分变量")
    customer_prompt = ChatPromptTemplate.from_messages(
        [("system", "你在为{brand}写文案，风格：{tone}。"), ("human", "为{product}写一句 15 字以内的 slogan。")]
    )
    # 品牌和语气固定，只留 product 给调用方
    brand_chain = customer_prompt.partial(brand="山野咖啡", tone="克制、有画面感") | model | StrOutputParser()
    print(f"  {brand_chain.invoke({'product': '冷萃黑咖啡'}).strip()}")


if __name__ == "__main__":
    main()
