"""第 4 章 · 示例 2：Agent 级结构化输出（response_format）

运行：
    uv run python chapters/ch04_structured/02_agent_structured.py

区别：
    模型级 with_structured_output —— 只有一次模型调用，模型"只输出结构"，不能调工具。
    Agent 级 response_format     —— 模型可以先调工具查数据，最后再按 schema 输出。

DeepSeek 专属坑（很重要）：
    1) 直接传 Pydantic 类，v1 会选 ProviderStrategy（用供应商原生 json_schema），
       DeepSeek 不支持 → 400 "This response_format type is unavailable now"。
    2) 改传 ToolStrategy，它会强制 tool_choice → 思考模式下 400。
    正确组合：reasoning_effort="none" + ToolStrategy(Schema)。
"""

from pydantic import BaseModel, Field

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.tools import tool

from langchain_learn import banner, get_model, section


@tool
def get_stock_price(symbol: str) -> str:
    """查询股票当前价格，symbol 例如 '600519'。"""
    fake = {"600519": "1680.5 元", "000001": "11.28 元"}
    return fake.get(symbol, f"{symbol}: 未找到")


class StockReport(BaseModel):
    """股票简报。"""

    symbol: str = Field(description="股票代码")
    name: str = Field(description="公司简称")
    price: str = Field(description="当前价格，带单位")
    trend: str = Field(description="一句话点评走势，20 字以内")


def main() -> None:
    banner("1. Agent + ToolStrategy：先查工具，再按 schema 输出")

    agent = create_agent(
        model=get_model(),  # 注意：必须是关闭思考的实例
        tools=[get_stock_price],
        system_prompt="你是证券助手，价格必须以工具返回为准。",
        response_format=ToolStrategy(StockReport),
    )

    result = agent.invoke({"messages": [{"role": "user", "content": "查一下 600519 现在多少钱"}]})

    structured = result["structured_response"]
    print(f"  structured_response 类型: {type(structured).__name__}")
    print(f"  内容: {structured}")
    print(f"  取字段: symbol={structured.symbol}, price={structured.price}")

    section("2. 同一次调用里，普通文本回答也还在 messages 中")
    print(f"  最后一条 AI 消息: {str(result['messages'][-1].content)[:80]}")
    print(f"  工具调用记录: {[type(m).__name__ for m in result['messages']]}")

    section("3. 踩坑复现：直接传 Pydantic 类会走 ProviderStrategy")
    try:
        bad_agent = create_agent(
            model=get_model(),
            tools=[get_stock_price],
            response_format=StockReport,  # 没包 ToolStrategy
        )
        bad_agent.invoke({"messages": [{"role": "user", "content": "查 600519"}]})
    except Exception as exc:  # noqa: BLE001
        print(f"  {type(exc).__name__}: {str(exc).splitlines()[0][:110]}")
        print("  结论：非 OpenAI 系模型上，显式写 ToolStrategy(Schema) 更稳。")

    section("4. 什么时候用哪个？")
    print("  只是把一段文本解析成对象            -> with_structured_output")
    print("  需要先查资料/调工具再给出结构化结论 -> create_agent + ToolStrategy")


if __name__ == "__main__":
    main()
