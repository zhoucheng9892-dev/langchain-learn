"""第 2 章 · 示例 1：三种创建模型的方式 + 关键参数

运行：
    uv run python chapters/ch02_models/01_model_init.py

LangChain 1.x 里模型有两种用法：
    A. 传字符串 "provider:model"  —— 配置驱动，适合换模型不改代码
    B. 传模型实例                —— 需要精细控制参数、或用非标准网关时用
两者在 create_agent / 链里可以混用。
"""

from langchain.chat_models import init_chat_model
from langchain_deepseek import ChatDeepSeek

from langchain_learn import MODEL_REF, banner, get_model, section


def main() -> None:
    banner("1. 方式 A：init_chat_model 用字符串引用")

    # 字符串写法 "deepseek:deepseek-flash" 会被解析成 provider=deepseek, model=deepseek-flash，
    # 然后自动 import langchain_deepseek 并实例化 ChatDeepSeek。
    model_a = init_chat_model(MODEL_REF, temperature=0, max_tokens=256)
    print(f"实例类型  : {type(model_a).__name__}")
    print(f"模型名    : {model_a.model_name}")
    print(f"温度      : {model_a.temperature}")

    section("2. 方式 B：直接实例化 ChatDeepSeek（参数最全）")
    model_b = ChatDeepSeek(
        model="deepseek-flash",  # 官方 API 只认这个 ID
        temperature=0,
        max_tokens=256,
        reasoning_effort="none",  # 关闭思考模式
        max_retries=3,  # SDK 层自动重试（网络/429）
        timeout=60,
    )
    print(f"实例类型  : {type(model_b).__name__}")
    print(f"重试/超时 : {model_b.max_retries} 次 / {model_b.request_timeout}s")

    section("3. 方式 C：项目统一的 get_model()（推荐）")
    model_c = get_model()
    print(f"实例类型  : {type(model_c).__name__}  模型: {model_c.model_name}")

    # ---- 参数怎么影响输出 ------------------------------------------------
    section("4. temperature：0 让结果稳定，1.3 让结果发散")
    prompt = "给一家只卖三种咖啡的小店起个名字，只输出名字本身。"
    for temp in (0.0, 1.3):
        m = get_model(temperature=temp, max_tokens=32)
        # 同一个 temperature 连跑两次，观察是否稳定
        names = [m.invoke(prompt).content.strip() for _ in range(2)]
        print(f"temperature={temp:<4} -> {names}")

    # ---- 常见错误：模型名写错 ------------------------------------------
    section("5. 常见错误演示：模型名不存在")
    try:
        bad = ChatDeepSeek(model="deepseek-v4.1-flash", max_tokens=16)
        bad.invoke("hi")
    except Exception as exc:  # noqa: BLE001
        detail = str(exc).split("\n")[0]
        print(f"报错类型: {type(exc).__name__}")
        print(f"报错信息: {detail[:180]}")
        print("结论    : 产品名不等于 API 模型名，统一走 get_model() 做映射。")


if __name__ == "__main__":
    main()
