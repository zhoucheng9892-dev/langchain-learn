"""第 6 章 · 示例 1：create_agent 的全部关键参数 + 执行轨迹

运行：
    uv run python chapters/ch06_agent/01_agent_anatomy.py

create_agent 的签名可以理解为：
    Agent = 模型 + 工具 + 系统提示词 + 状态 + 上下文 + 中间件 + 结构化输出
其余都是这七样的变体。这一节把每一个都过一遍。
"""

from pydantic import BaseModel, Field

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.tools import tool

from langchain_learn import MODEL_REF, banner, get_model, section


@tool
def search_flights(from_city: str, to_city: str, date: str) -> str:
    """查询两个城市之间某天的航班。日期格式 YYYY-MM-DD。"""
    return f"{from_city}->{to_city} {date}: CZ1234 08:30 起飞，¥1280；MU5678 19:10 起飞，¥990"


@tool
def get_weather(city: str) -> str:
    """查询城市天气。"""
    return f"{city}：多云，18°C"


class TripPlan(BaseModel):
    """行程建议。"""

    from_city: str = Field(description="出发城市")
    to_city: str = Field(description="到达城市")
    date: str = Field(description="日期 YYYY-MM-DD")
    cheapest_price: int = Field(description="最便宜航班价格，单位元")
    advice: str = Field(description="一句话出行建议，30 字以内")


def main() -> None:
    banner("1. 只给模型和工具（最小配置）")
    minimal = create_agent(model=MODEL_REF, tools=[get_weather])
    r = minimal.invoke({"messages": [{"role": "user", "content": "上海天气？"}]})
    print(f"  {r['messages'][-1].content.strip()[:60]}")

    section("2. 加上 system_prompt（决定行为边界）")
    with_prompt = create_agent(
        model=MODEL_REF,
        tools=[get_weather],
        system_prompt="你是气象台的播报员，用播音腔回答，一句话。",
    )
    r = with_prompt.invoke({"messages": [{"role": "user", "content": "上海天气？"}]})
    print(f"  {r['messages'][-1].content.strip()[:80]}")

    section("3. 加上 response_format（结构化收口）")
    full = create_agent(
        model=get_model(),  # 结构化输出需要关闭思考
        tools=[search_flights, get_weather],
        system_prompt=(
            "你是差旅助手。查航班必须调用 search_flights；"
            "如果目的地天气不好要在建议里提醒。"
        ),
        response_format=ToolStrategy(TripPlan),
        name="trip_planner",  # 作为子图被复用时有用
    )
    result = full.invoke(
        {"messages": [{"role": "user", "content": "帮我看看 11 月 5 日北京到成都的航班，顺便看看成都天气。"}]}
    )
    plan = result["structured_response"]
    print(f"  结构化结果: {plan}")

    section("4. 展开执行轨迹：每一步都发生了什么")
    for i, m in enumerate(result["messages"]):
        kind = type(m).__name__
        if getattr(m, "tool_calls", None):
            detail = ", ".join(f"{c['name']}({c['args']})" for c in m.tool_calls)
            print(f"    {i}. {kind:<13} 调工具 {detail}")
        elif kind == "ToolMessage":
            print(f"    {i}. {kind:<13} {m.content}")
        else:
            print(f"    {i}. {kind:<13} {str(m.content)[:52]}")

    section("5. 三个容易搞混的字段")
    print(f"  result['messages']            -> 完整对话，长度 {len(result['messages'])}")
    print(f"  result['structured_response'] -> {type(plan).__name__} 对象，只在配了 response_format 时存在")
    print("  result['jump_to']             -> 中间件做跳转控制时用，普通场景不出现")


if __name__ == "__main__":
    main()
