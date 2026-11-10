"""第 3 章 · 示例 1：消息体系与 content_blocks

运行：
    uv run python chapters/ch03_messages/01_messages.py

1.x 里对话的基本单位是"消息列表"，不是拼字符串。
四种核心消息：
    SystemMessage  设定角色与规则
    HumanMessage   用户输入
    AIMessage      模型输出（可能带 tool_calls）
    ToolMessage    工具执行结果（必须带 tool_call_id 对应某次调用）
"""

from langchain.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_learn import banner, get_model, section, show_message


def main() -> None:
    model = get_model(max_tokens=256)

    banner("1. 手写三条消息，交给模型")
    messages = [
        SystemMessage("你是一名只讲事实的技术编辑，回答不超过 30 字。"),
        HumanMessage("LangChain 1.x 用什么函数创建 Agent？"),
    ]
    reply = model.invoke(messages)
    show_message(reply)

    section("2. 消息对象里有什么")
    print(f"  type           = {type(reply).__name__}")
    print(f"  content        = {reply.content!r}")
    print(f"  content_blocks = {reply.content_blocks}")
    print(f"  id             = {reply.id}")
    print(f"  usage_metadata = {reply.usage_metadata}")

    section("3. 多轮对话：自己维护列表（无状态模型 + 有状态列表）")
    history = [SystemMessage("你是一个简洁的助手，每次回答不超过 20 字。")]
    for question in ["我叫小林，请记住。", "我叫什么名字？"]:
        history.append(HumanMessage(question))
        answer = model.invoke(history)
        history.append(answer)
        print(f"  用户: {question}")
        print(f"  模型: {answer.content.strip()}")
    print(f"  历史消息数: {len(history)}（模型本身不记任何东西，全靠这个列表）")

    section("4. ToolMessage 必须和 tool_call_id 对上")
    # 手动模拟一次工具调用：AI 发起调用 -> 我们执行 -> 把结果作为 ToolMessage 回填
    ai_with_call = AIMessage(
        content="",
        tool_calls=[{"name": "get_weather", "args": {"city": "杭州"}, "id": "call_demo_1"}],
    )
    tool_msg = ToolMessage(content="杭州：小雨，22°C", tool_call_id="call_demo_1", name="get_weather")
    print(f"  AI 发起: {ai_with_call.tool_calls[0]['name']}({ai_with_call.tool_calls[0]['args']})")
    print(f"  工具返回: {tool_msg.content}  (tool_call_id={tool_msg.tool_call_id})")

    follow_up = model.invoke([HumanMessage("杭州天气如何？"), ai_with_call, tool_msg])
    print(f"  模型据此作答: {follow_up.content.strip()}")

    section("5. 提示：可以直接用字典，LangChain 会自动转成消息对象")
    simple = model.invoke([{"role": "user", "content": "只回复两个字：收到"}])
    print(f"  {simple.content.strip()}")


if __name__ == "__main__":
    main()
