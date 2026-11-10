"""第 5 章 · 示例 1：工具的本质 + 手写一遍 Agent 循环

运行：
    uv run python chapters/ch05_tools/01_tool_basics.py

工具不是魔法：@tool 只是把函数变成"能被模型理解的 JSON Schema + 可执行对象"。
真正让 Agent 跑起来的是下面这段循环：
    while True:
        ai = model.invoke(messages)
        if not ai.tool_calls: break
        for call in ai.tool_calls: messages.append(tool.invoke(call))
create_agent 内部做的就是这个（外加状态、持久化、中间件）。
"""

from typing import Literal

from pydantic import BaseModel, Field

from langchain.messages import HumanMessage, SystemMessage
from langchain.tools import tool

from langchain_learn import banner, get_model, section, show_message


# ---- 1. 最简单的工具：函数签名 + docstring 就是说明书 --------------------
@tool
def get_weather(city: str) -> str:
    """查询指定城市的当前天气。"""
    return f"{city}：晴，25°C"


# ---- 2. 用 args_schema 精确描述参数（推荐） ------------------------------
class SearchInput(BaseModel):
    """搜索参数。"""

    query: str = Field(description="搜索关键词，尽量具体")
    top_k: int = Field(default=3, ge=1, le=10, description="返回条数")
    category: Literal["news", "docs", "blog"] = Field(default="docs", description="内容类型")


@tool("search_knowledge", args_schema=SearchInput)
def search_knowledge(query: str, top_k: int = 3, category: str = "docs") -> str:
    """在内部知识库中检索资料。"""
    return f"[{category}] 命中 {top_k} 条关于「{query}」的结果"


def main() -> None:
    banner("1. 工具对象长什么样")

    print(f"  name        : {get_weather.name}")
    print(f"  description : {get_weather.description}")
    print(f"  参数 schema : {get_weather.args_schema.model_json_schema()['properties']}")

    section("2. 复杂一点的参数：Pydantic args_schema")
    schema = search_knowledge.args_schema.model_json_schema()
    print(f"  必填: {schema.get('required')}")
    for name, prop in schema["properties"].items():
        print(f"    - {name}: {prop.get('type', prop.get('enum'))} / {prop.get('description')}")

    section("3. 直接调用工具（不经过模型）")
    print(f"  {get_weather.invoke({'city': '成都'})}")
    print(f"  {get_weather.invoke('成都')}   # 单参数可以直接传值")

    banner("4. 手写 Agent 循环：看清 create_agent 到底做了什么")

    model = get_model(max_tokens=512)
    tools = [get_weather, search_knowledge]
    tool_map = {t.name: t for t in tools}
    model_with_tools = model.bind_tools(tools)  # 把 schema 挂到模型上

    messages = [
        SystemMessage("你是助手。需要外部信息时必须调用工具。"),
        HumanMessage("成都天气怎么样？顺便在知识库里查一下 LangChain 中间件。"),
    ]

    for step in range(1, 6):  # 加个上限，防止死循环
        print(f"\n  --- 第 {step} 轮 ---")
        ai = model_with_tools.invoke(messages)
        show_message(ai)
        messages.append(ai)

        if not ai.tool_calls:
            print("\n  没有工具调用 -> 循环结束")
            break

        for call in ai.tool_calls:
            result = tool_map[call["name"]].invoke(call)  # 传整个 ToolCall，自动带上 id
            messages.append(result)
            print(f"    执行 {call['name']} -> {result.content}")

    section("5. 最终消息序列")
    for m in messages:
        print(f"  {type(m).__name__:<14} {str(m.content)[:50]}")

    print("\n  这 30 行就是 Agent。create_agent 帮你补上的是：")
    print("    状态管理、最大轮数、并行工具调用、错误处理、持久化、中间件钩子。")


if __name__ == "__main__":
    main()
