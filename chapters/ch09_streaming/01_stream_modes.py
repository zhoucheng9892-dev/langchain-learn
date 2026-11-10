"""第 9 章 · 示例 1：四种 stream_mode

运行：
    uv run python chapters/ch09_streaming/01_stream_modes.py

invoke 是一次性拿结果，stream 是"边跑边给"。
Agent 的流有四种粒度：
    updates  每个节点跑完后的增量状态（最常用，便于展示"正在调用工具"）
    values   每一步的完整状态（消息全量，便于前端整体重渲染）
    messages 模型输出的 token 级增量 + 工具消息（做打字机效果）
    custom   工具自己写的进度事件（做进度条）
"""

from langchain.agents import create_agent
from langchain.tools import ToolRuntime, tool

from langchain_learn import banner, get_model, section


@tool
def crawl_pages(topic: str, runtime: ToolRuntime) -> str:
    """抓取某个主题下的若干页面（演示 custom 流式进度）。"""
    writer = runtime.stream_writer
    pages = [f"{topic} 概览", f"{topic} 最佳实践", f"{topic} 常见坑"]
    for i, page in enumerate(pages, 1):
        writer({"progress": i / len(pages), "step": f"抓取 {page}"})  # custom 事件
    return f"共抓取 {len(pages)} 页：{'、'.join(pages)}"


def main() -> None:
    agent = create_agent(
        model=get_model(max_tokens=256),
        tools=[crawl_pages],
        system_prompt="你是研究助手，回答简短。",
    )
    question = "帮我抓一下 LangChain 中间件的资料，然后一句话总结。"
    inputs = {"messages": [{"role": "user", "content": question}]}

    banner("1. stream_mode='updates'：每个节点完成后的增量")
    for chunk in agent.stream(inputs, stream_mode="updates"):
        for node, update in chunk.items():
            msgs = update.get("messages", []) if isinstance(update, dict) else []
            kinds = [type(m).__name__ for m in msgs]
            print(f"  节点 {node:<8} -> {kinds}")

    section("2. stream_mode='values'：每一步的完整状态")
    step = 0
    for state in agent.stream(inputs, stream_mode="values"):
        step += 1
        print(f"  第 {step} 步: 共 {len(state['messages'])} 条消息，最后一条 {type(state['messages'][-1]).__name__}")

    section("3. stream_mode='messages'：token 级增量（打字机效果）")
    printing = False
    for msg_chunk, metadata in agent.stream(inputs, stream_mode="messages"):
        if metadata.get("langgraph_node") != "model":
            continue
        if msg_chunk.content:
            if not printing:
                print("  助手: ", end="")
                printing = True
            print(msg_chunk.content, end="", flush=True)
    print()

    section("4. stream_mode='custom'：工具自己上报进度")
    for item in agent.stream(inputs, stream_mode="custom"):
        print(f"  进度 {item['progress'] * 100:.0f}%  {item['step']}")

    section("5. 同时订阅多种模式（元组）")
    for mode, payload in agent.stream(inputs, stream_mode=["updates", "custom"]):
        if mode == "custom":
            print(f"  [custom] {payload['step']}")
        else:
            print(f"  [updates] 节点 {list(payload.keys())}")


if __name__ == "__main__":
    main()
