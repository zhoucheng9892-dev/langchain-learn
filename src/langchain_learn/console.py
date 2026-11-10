"""控制台输出小工具：让示例输出分段清晰、便于对照文档阅读。

刻意只用 ASCII 边框字符，避免 Windows 终端编码问题。
"""

from __future__ import annotations

import textwrap

WIDTH = 72


def hr(char: str = "-") -> None:
    print(char * WIDTH)


def banner(text: str) -> None:
    print()
    print("=" * WIDTH)
    print(f"  {text}")
    print("=" * WIDTH)


def section(text: str) -> None:
    print()
    print(f"[{text}]")
    hr()


def kv(key: str, value: object, indent: int = 2) -> None:
    print(f"{' ' * indent}{key}: {value}")


def show_message(msg, *, max_len: int = 400) -> None:
    """打印一条消息的类型与内容（含工具调用 / 思考内容）。"""
    name = type(msg).__name__
    content = msg.content
    if isinstance(content, list):
        # 标准内容块（content_blocks）在 v1 里是列表形式
        content = " ".join(
            str(block.get("text", block)) if isinstance(block, dict) else str(block)
            for block in content
        )
    text = str(content)
    if len(text) > max_len:
        text = text[:max_len] + " ...(截断)"
    print(f"  <{name}> {text}")

    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls:
        for tc in tool_calls:
            print(f"      -> tool_call: {tc['name']}({tc['args']})")

    reasoning = (getattr(msg, "additional_kwargs", {}) or {}).get("reasoning_content")
    if reasoning:
        preview = textwrap.shorten(str(reasoning).replace("\n", " "), width=120)
        print(f"      (思考) {preview}")


def show_messages(messages) -> None:
    for msg in messages:
        show_message(msg)
