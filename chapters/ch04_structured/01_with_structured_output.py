"""第 4 章 · 示例 1：模型级结构化输出 with_structured_output

运行：
    uv run python chapters/ch04_structured/01_with_structured_output.py

把"让模型返回一段 JSON 再自己 json.loads"这件脏活交给 LangChain：
传入 Pydantic 模型，拿回来的是已经校验过的对象。

DeepSeek 注意点：
    v4 系列默认开启思考模式，而思考模式不支持强制 tool_choice，
    with_structured_output 默认走 function_calling（会强制 tool_choice）→ 直接 400。
    解决办法：reasoning_effort="none"（get_model 默认已经帮你关掉了）。
"""

from enum import Enum

from pydantic import BaseModel, Field

from langchain_learn import banner, get_model, section


# ---- 1. 最简单的 schema ------------------------------------------------
class Movie(BaseModel):
    """一部电影的基本信息。"""

    title: str = Field(description="电影中文名")
    year: int = Field(description="上映年份，四位数字")
    rating: float = Field(description="豆瓣评分，0-10")


# ---- 2. 带枚举和约束的 schema ------------------------------------------
class Sentiment(str, Enum):
    positive = "positive"
    neutral = "neutral"
    negative = "negative"


class Review(BaseModel):
    """一条商品评论的结构化结果。"""

    sentiment: Sentiment = Field(description="整体情感倾向")
    score: int = Field(ge=1, le=5, description="1-5 星")
    pros: list[str] = Field(default_factory=list, description="优点，最多 3 条")
    cons: list[str] = Field(default_factory=list, description="缺点，最多 3 条")
    would_recommend: bool = Field(description="是否推荐给朋友")


def main() -> None:
    model = get_model()  # 思考已关闭，可以直接用强制 tool_choice

    banner("1. 基础用法：Pydantic in，Pydantic out")
    structured = model.with_structured_output(Movie)
    movie = structured.invoke("推荐一部 2020 年之后的高分科幻电影。")
    print(f"  类型: {type(movie).__name__}")
    print(f"  内容: {movie}")
    print(f"  取字段: movie.title = {movie.title!r}, movie.rating = {movie.rating}")

    section("2. 枚举 + 数值约束：把'合法值'交给 schema 管")
    review_parser = model.with_structured_output(Review)
    review = review_parser.invoke(
        "评价这条评论：「耳机音质不错，降噪也够用，就是戴久了有点夹耳朵，价格偏贵。」"
    )
    print(f"  情感: {review.sentiment.value}   星级: {review.score}   推荐: {review.would_recommend}")
    print(f"  优点: {review.pros}")
    print(f"  缺点: {review.cons}")

    section("3. 嵌套结构：列表里再套对象")
    class Chapter(BaseModel):
        title: str
        summary: str = Field(description="不超过 20 字")

    class Outline(BaseModel):
        topic: str
        chapters: list[Chapter] = Field(description="3 个章节")

    outline = model.with_structured_output(Outline).invoke("给「LangChain 入门」写一个 3 章大纲。")
    print(f"  主题: {outline.topic}")
    for i, ch in enumerate(outline.chapters, 1):
        print(f"    {i}. {ch.title} — {ch.summary}")

    section("4. include_raw=True：同时拿到原始消息，便于排查解析失败")
    raw_parser = model.with_structured_output(Movie, include_raw=True)
    result = raw_parser.invoke("说一部 1994 年的经典电影。")
    print(f"  keys       = {list(result.keys())}")
    print(f"  parsed     = {result['parsed']}")
    print(f"  parsing_error = {result['parsing_error']}")
    print(f"  raw 类型   = {type(result['raw']).__name__}")

    section("5. 踩坑复现：思考模式下做结构化输出会失败")
    thinking_model = get_model(thinking=True, max_tokens=1024)
    try:
        thinking_model.with_structured_output(Movie).invoke("推荐一部电影")
    except Exception as exc:  # noqa: BLE001
        print(f"  {type(exc).__name__}: {str(exc).splitlines()[0][:120]}")
        print("  结论：结构化输出请用 get_model()（reasoning_effort='none'）。")


if __name__ == "__main__":
    main()
