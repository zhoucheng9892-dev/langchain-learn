"""第 8 章 · 示例 2：长期记忆（Store）—— 跨会话记住"这个人是谁"

运行：
    uv run python chapters/ch08_memory/02_long_term_memory.py

两种记忆别搞混：
    checkpointer（短期记忆）：一次会话的对话历史，按 thread_id 隔离，会话结束就该消失。
    store       （长期记忆）：跨会话的事实，按 namespace + key 存，永久保留。

判断标准：下次用户换个话题、隔一周再来，这条信息还该不该在？
    该在   -> store（姓名、口味、套餐、历史订单）
    不该在 -> checkpointer（"我刚才说到哪了"）
"""

from dataclasses import dataclass

from langgraph.store.memory import InMemoryStore

from langchain.agents import create_agent
from langchain.tools import ToolRuntime, tool

from langchain_learn import banner, get_model, section


@dataclass
class UserContext:
    user_id: str


# ---- store 里的"表"叫 namespace，用元组表示；key 是主键 -----------------
NS = ("preferences",)  # 真实项目里常写成 ("preferences", user_id)


@tool
def save_preference(key: str, value: str, runtime: ToolRuntime[UserContext]) -> str:
    """记住用户的一条长期偏好，例如 key='咖啡' value='不加糖'。"""
    user_id = runtime.context.user_id
    runtime.store.put((*NS, user_id), key, {"value": value})
    return f"已记住：{key} = {value}"


@tool
def recall_all_preferences(runtime: ToolRuntime[UserContext]) -> str:
    """列出该用户所有已记住的偏好。"""
    user_id = runtime.context.user_id
    items = runtime.store.search((*NS, user_id))
    if not items:
        return "还没有记录任何偏好。"
    return "；".join(f"{it.key}={it.value['value']}" for it in items)


def build(store):
    return create_agent(
        model=get_model(max_tokens=384),
        tools=[save_preference, recall_all_preferences],
        system_prompt=(
            "你是咖啡店点单助手。用户透露偏好时必须调用 save_preference 记住；"
            "用户点单前先调用 recall_all_preferences 查已有偏好。回答 40 字以内。"
        ),
        store=store,
        context_schema=UserContext,
    )


def chat(agent, text: str, user_id: str, thread_id: str) -> str:
    r = agent.invoke(
        {"messages": [{"role": "user", "content": text}]},
        config={"configurable": {"thread_id": thread_id}},
        context=UserContext(user_id=user_id),
    )
    return str(r["messages"][-1].content).strip()


def main() -> None:
    store = InMemoryStore()  # 生产用 PostgresStore / RedisStore，接口一致
    agent = build(store)

    banner("1. 第一次会话：用户随口说了自己的口味")
    print(f"  用户: 我喝咖啡不加糖，喜欢燕麦奶。")
    print(f"  助手: {chat(agent, '我喝咖啡不加糖，喜欢燕麦奶。', 'u_001', 'thread-day1')}")

    section("2. store 里到底存了什么")
    for item in store.search(("preferences", "u_001")):
        print(f"  key={item.key!r} value={item.value} namespace={item.namespace}")

    section("3. 换一个 thread（等于换了一天/换了个话题），记忆还在")
    print(f"  用户: 老样子，来一杯。")
    print(f"  助手: {chat(agent, '老样子，来一杯。', 'u_001', 'thread-day7')}")

    section("4. 换一个用户：完全隔离")
    print(f"  用户(u_002): 来杯美式。")
    print(f"  助手: {chat(agent, '来杯美式。', 'u_002', 'thread-u2')}")
    print(f"  u_002 的偏好: {[it.key for it in store.search(('preferences', 'u_002'))] or '（空）'}")

    section("5. 三种记忆的分工")
    print("  messages（本次会话）  -> checkpoint，按 thread_id")
    print("  store（跨会话事实）   -> 本例，按 namespace + key")
    print("  文件/向量库（知识）   -> 第 10 章 RAG")
    print("  注意：store 不会自动进模型上下文，必须由工具或中间件主动读出来。")


if __name__ == "__main__":
    main()
