"""第 12 章 · 示例 3：成本与缓存（把账单按下去）

运行：
    uv run python chapters/ch12_production/03_cost_and_cache.py

三条最有效的省钱手段，按性价比排序：
    1. 缓存：完全相同的问题不重复付钱（前缀缓存 / 精确缓存）；
    2. 缩短上下文：少放无关历史，用摘要代替全量；
    3. 降低输出：max_tokens 设上限，提示词要求简短。

注意：缓存对"Temperature=0 + 相同输入"才有意义；开放闲聊场景命中率很低。
"""

import time

from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache

from langchain_learn import banner, get_model, section

# 价格按百万 token 计（示例值，请替换成你的实际报价）
PRICE_IN = 0.5   # 元 / 百万输入 token
PRICE_OUT = 2.0  # 元 / 百万输出 token


def cost_of(usage: dict) -> float:
    return usage["input_tokens"] / 1e6 * PRICE_IN + usage["output_tokens"] / 1e6 * PRICE_OUT


def main() -> None:
    long_context = "以下是产品手册节选，请基于它回答。\n" + ("LangChain 是一个用于构建大模型应用的框架。" * 60)
    question = "用一句话概括这段材料在讲什么。"

    banner("1. 没有缓存：同一个问题问两遍，付两次钱")
    model = get_model(max_tokens=96)
    t0 = time.perf_counter()
    r1 = model.invoke(f"{long_context}\n\n问题：{question}")
    t1 = time.perf_counter()
    r2 = model.invoke(f"{long_context}\n\n问题：{question}")
    t2 = time.perf_counter()
    print(f"  第 1 次: {(t1 - t0) * 1000:.0f}ms  token={r1.usage_metadata['total_tokens']}  ¥{cost_of(r1.usage_metadata):.6f}")
    print(f"  第 2 次: {(t2 - t1) * 1000:.0f}ms  token={r2.usage_metadata['total_tokens']}  ¥{cost_of(r2.usage_metadata):.6f}")
    print(f"  两次合计: ¥{cost_of(r1.usage_metadata) + cost_of(r2.usage_metadata):.6f}")

    section("2. 打开缓存：第二遍直接命中，0 token 0 费用")
    set_llm_cache(InMemoryCache())
    t0 = time.perf_counter()
    c1 = model.invoke(f"{long_context}\n\n问题：{question}")
    t1 = time.perf_counter()
    c2 = model.invoke(f"{long_context}\n\n问题：{question}")
    t2 = time.perf_counter()
    print(f"  第 1 次(未命中): {(t1 - t0) * 1000:.0f}ms  token={c1.usage_metadata['total_tokens']}")
    print(f"  第 2 次(命中)  : {(t2 - t1) * 1000:.1f}ms  token={c2.usage_metadata['total_tokens']}")
    print(f"  命中后耗时下降 {(t1 - t0) / max(t2 - t1, 1e-6):.0f} 倍")
    print("  c2 与 c1 内容是否一致:", c1.content == c2.content)

    section("3. 缓存的边界")
    print("  精确缓存：输入字符串完全一致才命中（本例）")
    print("  语义缓存：向量相似度达到阈值即命中（要自己实现，注意误命中风险）")
    print("  供应商前缀缓存：DeepSeek / OpenAI 自动对相同前缀打折，不改变结果语义")
    print("  生产建议：本地用 SqliteCache 或 RedisCache，多实例共享")

    section("4. 成本估算表（按本项目的实际用量）")
    scenarios = [
        ("单轮问答", 800, 120, 1),
        ("RAG 问答(3 段上下文)", 3000, 200, 1),
        ("Agent 两轮工具", 2400, 300, 1),
        ("客服机器人 1 万次/天(多轮平均)", 4000, 250, 10000),
    ]
    print(f"  {'场景':<28}{'输入':>8}{'输出':>8}{'次数':>8}{'费用':>12}")
    for name, tin, tout, times in scenarios:
        c = (tin / 1e6 * PRICE_IN + tout / 1e6 * PRICE_OUT) * times
        print(f"  {name:<28}{tin:>8}{tout:>8}{times:>8}{'¥' + format(c, '.4f'):>12}")
    print("  结论：真正吃掉预算的是「每天多少次」，不是「单次多贵」。")

    section("5. 组合拳")
    print("  缓存命中 + 上下文裁剪 + max_tokens 上限 + 小模型分流（简单问题走 flash）")
    print("  这四件事做完，同等业务量下成本通常能降 50%~80%。")


if __name__ == "__main__":
    main()
