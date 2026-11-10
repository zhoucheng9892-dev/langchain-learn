"""第 2 章 · 示例 3：异步与并发（生产环境必须掌握）

运行：
    uv run python chapters/ch02_models/03_async_concurrency.py

同步 for 循环串行调用 5 次 = 5 倍等待时间。
异步 + 并发把等待时间重叠起来，同样的钱、更短的时间。

注意：一个进程里只调用一次 asyncio.run()，所有异步任务都在这一个事件循环里完成。
在 Windows 上反复 asyncio.run() 复用同一个 httpx 客户端，会在退出时报
"Event loop is closed"，这是常见坑，不是你的代码写错了。
"""

import asyncio
import time

from langchain_learn import banner, get_model, section

QUESTIONS = [
    "用一句话解释什么是向量数据库。",
    "用一句话解释什么是 function calling。",
    "用一句话解释什么是 prompt engineering。",
    "用一句话解释什么是上下文窗口。",
    "用一句话解释什么是 token。",
]


async def one(model, question: str) -> str:
    msg = await model.ainvoke(question)
    return msg.content.strip()


async def main_async() -> None:
    model = get_model(max_tokens=128)

    banner("1. 串行：5 个问题一个一个问")
    start = time.perf_counter()
    for q in QUESTIONS:
        await one(model, q)
    serial_cost = time.perf_counter() - start
    print(f"耗时: {serial_cost:.2f}s")

    section("2. 并发：asyncio.gather 一起问")
    start = time.perf_counter()
    results = await asyncio.gather(*(one(model, q) for q in QUESTIONS))
    concurrent_cost = time.perf_counter() - start
    print(f"耗时: {concurrent_cost:.2f}s   提速 {serial_cost / concurrent_cost:.1f}x")

    section("3. 限流：用 Semaphore 控制同时在飞的请求数")
    sem = asyncio.Semaphore(2)  # 最多 2 个请求同时在飞，避免触发供应商限流

    async def limited(q: str) -> str:
        async with sem:
            return await one(model, q)

    start = time.perf_counter()
    limited_results = await asyncio.gather(*(limited(q) for q in QUESTIONS))
    print(f"耗时: {time.perf_counter() - start:.2f}s（比全并发慢，但生产上更稳）")
    for text in limited_results:
        print(f"  - {text[:44]}")

    section("4. 异步流式 astream")
    print("  ", end="")
    async for chunk in model.astream("用三个词形容 LangChain 1.x 的设计风格。"):
        print(chunk.content, end="", flush=True)
    print()

    section("5. 结论")
    print("  能用异步就用异步：Web 服务、批量任务、Agent 并行工具调用都是异步内部实现。")
    print("  并发数要有上限，配合 max_retries 处理偶发 429。")


if __name__ == "__main__":
    asyncio.run(main_async())
