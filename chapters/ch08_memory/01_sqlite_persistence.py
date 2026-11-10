"""第 8 章 · 示例 1：跨进程的短期记忆（SQLite checkpointer）

运行：
    uv run python chapters/ch08_memory/01_sqlite_persistence.py           # 第一次：写入
    uv run python chapters/ch08_memory/01_sqlite_persistence.py --read    # 第二次：读回

InMemorySaver 一关进程就没了。SqliteSaver 把 checkpoint 落到磁盘，
于是"用户今天问过的事，明天还能接上"。
生产上通常是 PostgresSaver，接口一样。
"""

import sqlite3
import sys
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

from langchain.agents import create_agent
from langchain.messages import HumanMessage

from langchain_learn import banner, get_model, section

DB_PATH = Path(__file__).resolve().parents[2] / ".data" / "checkpoints.db"
DB_PATH.parent.mkdir(exist_ok=True)
THREAD = "user-42-support"


def build_agent():
    # check_same_thread=False：允许 LangGraph 在不同线程里使用同一个连接
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    return create_agent(
        model=get_model(max_tokens=256),
        tools=[],
        system_prompt="你是售后助手，回答简洁（30 字以内）。",
        checkpointer=SqliteSaver(conn),
    )


def main() -> None:
    agent = build_agent()
    cfg = {"configurable": {"thread_id": THREAD}}

    if "--read" in sys.argv:
        banner("第二次运行：新进程，只读回历史，不提新问题")
        state = agent.get_state(cfg)
        messages = state.values.get("messages", [])
        print(f"  从 {DB_PATH.name} 读回 {len(messages)} 条消息：")
        for m in messages:
            print(f"    {type(m).__name__:<13} {str(m.content)[:60]}")
        if not messages:
            print("  （空）。请先不带 --read 运行一次。")
        return

    banner("第一次运行：聊两句，写入磁盘")
    r1 = agent.invoke(
        {"messages": [HumanMessage("我上周买的耳机左耳没声音，订单号是 A2024。")]}, config=cfg
    )
    print(f"  用户: 我上周买的耳机左耳没声音，订单号是 A2024。")
    print(f"  助手: {r1['messages'][-1].content.strip()}")

    r2 = agent.invoke({"messages": [HumanMessage("我想怎么处理来着？")]}, config=cfg)
    print(f"  用户: 我想怎么处理来着？")
    print(f"  助手: {r2['messages'][-1].content.strip()}")

    section("已落盘")
    print(f"  数据库: {DB_PATH}")
    print(f"  线程  : {THREAD}")
    print("  现在关掉进程，再运行一次并加上 --read，历史还在。")

    section("换成 Postgres（生产常用）")
    print("  from langgraph.checkpoint.postgres import PostgresSaver")
    print("  with PostgresSaver.from_conn_string(DSN) as saver:")
    print("      saver.setup()")
    print("      agent = create_agent(..., checkpointer=saver)")


if __name__ == "__main__":
    main()
