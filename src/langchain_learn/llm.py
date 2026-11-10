"""模型工厂：全项目统一从这里拿模型实例。

为什么需要这一层？
1. DeepSeek 官方 API 的模型名和产品名不完全一致（见下方的别名表），
   写死模型名很容易在换环境时炸掉。
2. v1 的 create_agent / with_structured_output 都受「思考模式」影响，
   把开关收敛到一处，示例代码才不会到处复制粘贴参数。
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

# 产品名 -> API 模型名。
# 有的平台（比如 IDE 内部）把 v4.1-flash 当作展示名，但 OpenAI 兼容接口
# 只认 deepseek-flash，传错会报 400：The supported API model names are ...
_MODEL_ALIASES = {
    "deepseek-v4.1-flash": "deepseek-flash",
    "deepseek-v4.1-pro": "deepseek-v4-pro",
    "deepseek-v4-flash": "deepseek-flash",
    "deepseek-v4-pro": "deepseek-v4-pro",
}

_model_name = os.getenv("MODEL_NAME", "deepseek-flash").strip()
_API_MODEL = _MODEL_ALIASES.get(_model_name, _model_name)

# 直接用字符串引用模型时（create_agent(model=MODEL_REF)）的写法
MODEL_REF = f"deepseek:{_API_MODEL}"


def get_model(
    *,
    thinking: bool = False,
    temperature: float = 0.0,
    max_tokens: int | None = 2048,
    **kwargs,
) -> ChatDeepSeek:
    """返回一个 ChatDeepSeek 实例。

    Args:
        thinking: 是否开启思考模式（v4 系列默认会输出 reasoning_content）。
            关闭后：响应更快、更省 token，并且可以用强制 tool_choice
            （结构化输出的 ToolStrategy 依赖它）。
        temperature: 采样温度，示例默认 0，保证可复现。
        max_tokens: 输出上限。思考模式会消耗输出额度，太小时答案会被截断。
    """
    params: dict = {
        "model": _API_MODEL,
        "temperature": temperature,
        # reasoning_effort="none" 是关闭思考的官方开关；
        # 传 "high" 可以显式要求更长的思考。
        "reasoning_effort": "high" if thinking else "none",
    }
    if max_tokens is not None:
        params["max_tokens"] = max_tokens
    params.update(kwargs)
    return ChatDeepSeek(**params)


__all__ = ["MODEL_REF", "get_model", "ROOT"]
