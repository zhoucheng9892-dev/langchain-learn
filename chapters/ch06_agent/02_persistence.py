"""第 6 章 · 示例 2：多轮对话与持久化（checkpointer + thread_id）

运行：
    uv run python chapters/ch06_agent/02_persistence.py

模型是无状态的，Agent 默认也是。
要让它"记住上一轮"，必须挂 checkpointer，并在每次调用时带上同一个 thread_id。
    thread_id  = 一条会话（像微信里的一个聊天窗口）
    checkpointer = 存这些会话的地方（内存 / SQLite / Postgres）
"""

from langgraph.checkpoint.memory import InMemorySaver

from langchain.agents import create_agent
from langchain.tools import tool

from langchain_learn import banner, get_model, section


@tool
def remember_fact(fact: str) -> str:
    """把用户提供的事实记录到会话里。"""
    return f"已记录：{fact}"


def ask(agent, text: str, thread_id: str) -> str:
    """带上 thread_id 提问。"""
    result = agent.invoke(
        {"messages": [{"role": "user", "content": text}]},
        config={"configurable": {"thread_id": thread_id}},
    )
    return str(result["messages"][-1].content).strip()


def main() -> None:
    banner("1. 没挂 checkpointer：每轮都是失忆的")

    forgetful = create_agent(model=get_model(max_tokens=256), tools=[])
    print(f"  第 1 轮: {ask(forgetful, '我叫林墨，记住我的名字。', 't1')[:40]}")
    print(f"  第 2 轮: {ask(forgetful, '我叫什么？', 't1')[:40]}")
    print("  -> 它不知道。因为没有地方存历史。")

    section("2. 挂上 InMemorySaver：同一 thread_id 自动带上历史")
    agent = create_agent(
        model=get_model(max_tokens=256),
        tools=[remember_fact],
        system_prompt="你是一个有记忆的助手，回答简短。",
        checkpointer=InMemorySaver(),  # 关键：持久化后端
    )

    thread_a = "session-A"
    print(f"  [A] 第 1 轮: {ask(agent, '我叫林墨，是数据工程师。', thread_a)[:50]}")
    print(f"  [A] 第 2 轮: {ask(agent, '我叫什么？做什么工作？', thread_a)[:60]}")

    section("3. 换一个 thread_id：立刻是另一段独立会话")
    thread_b = "session-B"
    print(f"  [B] 第 1 轮: {ask(agent, '我叫什么？', thread_b)[:60]}")
    print("  -> 两个 thread 的历史完全隔离。")

    section("4. 读回某个 thread 的当前状态")
    state = agent.get_state({"configurable": {"thread_id": thread_a}})
    print(f"  消息数: {len(state.values['messages'])}")
    for m in state.values["messages"]:
        print(f"    {type(m).__name__:<13} {str(m.content)[:44]}")
    print(f"  下一个待执行节点: {state.next}")

    section("5. 生产环境：把 InMemorySaver 换成 SQLite / Postgres")
    print("  from langgraph.checkpoint.sqlite import SqliteSaver")
    print("  import sqlite3")
    print("  conn = sqlite3.connect('.data/checkpoints.db', check_same_thread=False)")
    print("  agent = create_agent(..., checkpointer=SqliteSaver(conn))")
    print("  # 进程重启后，同一个 thread_id 依然能接上上下文")
    print("  第 8 章会用 SQLite 真正跑一遍跨进程记忆。")


if __name__ == "__main__":
    main()
