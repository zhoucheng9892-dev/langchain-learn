"""第 9 章 · 示例 2：把流接到前端（SSE 最小实现）+ v3 事件协议

运行：
    uv run python chapters/ch09_streaming/02_sse_and_events.py

真实的 Web 应用里，后端通常把 chunk 转成 Server-Sent Events：
    data: {"type": "token", "text": "你"}\n\n
前端用 EventSource 或 fetch + ReadableStream 读。
这个示例不启服务，直接把要发的字节打印出来——你把它塞进
FastAPI 的 StreamingResponse 就能用。
"""

import json
import warnings

from langchain.agents import create_agent
from langchain.tools import tool

from langchain_learn import banner, get_model, section

warnings.filterwarnings("ignore")  # stream_events v3 目前是实验特性，会打 beta 警告


@tool
def get_time(city: str) -> str:
    """查询城市的当前时间。"""
    return f"{city} 当前时间 20:31"


def sse(event: dict) -> str:
    """把一条事件编码成 SSE 帧。"""
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def main() -> None:
    agent = create_agent(
        model=get_model(max_tokens=256),
        tools=[get_time],
        system_prompt="你是助手，回答一句话。",
    )
    inputs = {"messages": [{"role": "user", "content": "成都现在几点？顺便说句晚安。"}]}

    banner("1. 后端要把这些帧发给前端")
    for mode, payload in agent.stream(inputs, stream_mode=["messages", "updates"]):
        if mode == "messages":
            chunk, meta = payload
            if chunk.content and meta.get("langgraph_node") == "model":
                print(sse({"type": "token", "text": chunk.content}), end="")
        else:
            for node, update in payload.items():
                if node == "tools":
                    print(sse({"type": "tool", "name": "get_time", "status": "done"}), end="")

    section("2. 前端大概这么写")
    print(
        """
  const res = await fetch('/chat', {method: 'POST', body: JSON.stringify({q})});
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  while (true) {
    const {value, done} = await reader.read();
    if (done) break;
    for (const line of decoder.decode(value).split('\\n\\n')) {
      if (!line.startsWith('data: ')) continue;
      const ev = JSON.parse(line.slice(6));
      if (ev.type === 'token') appendText(ev.text);
      if (ev.type === 'tool') showToolBadge(ev.name);
    }
  }
"""
    )

    section("3. stream_events(version='v3')：结构化事件协议（实验特性）")
    print("  比 stream_mode 更细，事件带 namespace、run_id，适合做多 Agent 追溯。")
    stream = agent.stream_events(
        {"messages": [{"role": "user", "content": "成都几点？"}]}, version="v3"
    )
    seen = 0
    for event in stream:
        method = event.get("method")
        params = event.get("params", {})
        print(f"    method={method:<12} namespace={params.get('namespace')}")
        seen += 1
        if seen >= 8:
            print("    ...（事件很多，这里只列前 8 条）")
            break
    print("  可用访问器: stream.messages / .values / .tool_calls / .output / .lifecycle")

    section("4. 该用哪个？")
    print("  Web 打字机            -> stream_mode='messages'")
    print("  前端步骤条/工具状态    -> stream_mode='updates'")
    print("  工具上报进度           -> stream_mode='custom'")
    print("  多 Agent 全链路追溯    -> stream_events(version='v3')（注意是实验 API）")


if __name__ == "__main__":
    main()
