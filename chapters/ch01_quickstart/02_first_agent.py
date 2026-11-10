"""第 1 章 · 示例 2：第一个 Agent（30 行以内）

运行：
    uv run python chapters/ch01_quickstart/02_first_agent.py

对比示例 1：模型只会"说"。要让它"做"——查天气、算数、读数据库——
就必须给它工具，并且让它在"模型 -> 工具 -> 模型"的循环里自己决定调哪个。
create_agent 就是把这条循环封装好的函数。
"""

from langchain.agents import create_agent
from langchain.tools import tool

from langchain_learn import MODEL_REF, banner, section


# ---- 工具：一个普通的 Python 函数 + 文档字符串 -------------------------
# @tool 会把函数签名和 docstring 转成 JSON Schema 发给模型，
# 所以 docstring 不是给人看的注释，而是"给模型看的说明书"，必须写清用途。
@tool
def get_weather(city: str) -> str:
    """查询指定城市的当前天气。"""
    # 真实项目里这里应该是 HTTP 请求，示例用假数据保证离线可跑。
    fake_data = {
        "北京": "晴，25°C，西北风 3 级",
        "上海": "多云，28°C，湿度 70%",
        "深圳": "雷阵雨，31°C",
    }
    return fake_data.get(city, f"{city}：暂无数据")


@tool
def calculate(expression: str) -> str:
    """计算一个四则运算表达式，例如 '12*(3+4)'。"""
    # 只做演示：生产环境不要直接 eval 用户输入。
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as exc:  # noqa: BLE001
        return f"计算失败：{exc}"


def main() -> None:
    banner("1. 创建 Agent：模型 + 工具 + 系统提示词")

    agent = create_agent(
        # 直接传 "deepseek:deepseek-flash" 这样的字符串也可以，
        # MODEL_REF 只是把这个写法收敛到一处。
        model=MODEL_REF,
        tools=[get_weather, calculate],
        system_prompt=(
            "你是一个严谨的助手。需要外部数据或计算时必须调用工具，"
            "不要凭空编造结果。回答控制在两句话以内。"
        ),
    )

    # ---- 输入是一个 dict，必须有 messages 键 --------------------------
    section("2. 提问：北京天气怎么样？另外帮我算 (128+64)*3")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "北京天气怎么样？另外帮我算一下 (128+64)*3"}]}
    )

    # ---- 输出也是 dict，messages 里是完整的过程记录 --------------------
    # 一条典型的记录是：Human -> AI(带 tool_calls) -> Tool -> AI(最终回答)
    section("3. 完整执行轨迹（这就是 Agent 的思考过程）")
    for i, msg in enumerate(result["messages"]):
        kind = type(msg).__name__
        if getattr(msg, "tool_calls", None):
            calls = ", ".join(f"{c['name']}({c['args']})" for c in msg.tool_calls)
            print(f"{i}. {kind:<12} -> 调用工具：{calls}")
        elif kind == "ToolMessage":
            print(f"{i}. {kind:<12} -> 工具返回：{msg.content}")
        else:
            text = str(msg.content).replace("\n", " ")
            print(f"{i}. {kind:<12} -> {text[:60]}")

    section("4. 最终回答")
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
