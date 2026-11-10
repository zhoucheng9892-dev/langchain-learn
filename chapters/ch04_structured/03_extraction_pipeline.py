"""第 4 章 · 示例 3：把结构化输出用在真实流水线上（信息抽取 + 校验 + 兜底）

运行：
    uv run python chapters/ch04_structured/03_extraction_pipeline.py

真实业务里，结构化输出很少"一次就完美"。这个示例给出三段式套路：
    1. schema 里写清约束（枚举、范围、必填），让模型少犯错；
    2. 用 Pydantic validator 兜住业务规则（模型不懂你的业务口径）；
    3. 逐条处理 + 失败兜底，一批数据里坏一条不能炸整批。
"""

from enum import Enum

from pydantic import BaseModel, Field, ValidationError, field_validator

from langchain_learn import banner, get_model, section


class Level(str, Enum):
    junior = "初级"
    middle = "中级"
    senior = "高级"


class JobPost(BaseModel):
    """一条招聘信息。"""

    # 关键设计：先让模型判断"这段文本到底是不是招聘信息"。
    # 否则它会硬把无关文本套进 schema，编出"未知公司 / 月薪 1k"这种垃圾数据。
    is_job_post: bool = Field(description="这段文本是否真的是招聘信息")
    company: str = Field(default="", description="公司名，不是招聘信息时留空")
    title: str = Field(default="", description="职位名称，不是招聘信息时留空")
    city: str = Field(default="", description="工作城市，不是招聘信息时留空")
    level: Level = Field(default=Level.junior, description="职级")
    salary_min: int = Field(default=1, ge=1, le=500, description="月薪下限，单位：千元")
    salary_max: int = Field(default=1, ge=1, le=500, description="月薪上限，单位：千元")
    skills: list[str] = Field(default_factory=list, description="必备技能，最多 4 个")

    @field_validator("salary_max")
    @classmethod
    def check_range(cls, v: int, info):
        """业务规则：上限不能低于下限。模型经常把顺序写反。"""
        low = info.data.get("salary_min")
        if low is not None and v < low:
            raise ValueError(f"salary_max({v}) 不能小于 salary_min({low})")
        return v


RAW_POSTS = [
    "字节跳动招高级后端工程师，base 北京，月薪 40-70k，要求熟悉 Go、Kubernetes、分布式系统。",
    "某创业公司招前端，杭州，应届初级，15-25k，会 React 和 TypeScript 就行。",
    "这是一条无效信息：今天天气不错。",
]


def extract(text: str) -> JobPost | None:
    """单条抽取，失败返回 None。"""
    parser = get_model(max_tokens=512).with_structured_output(JobPost)
    try:
        job = parser.invoke(f"从下面这段文案里抽取结构化信息：\n{text}")
    except ValidationError as exc:
        print(f"    校验失败: {exc.errors()[0]['msg']}")
        return None
    except Exception as exc:  # noqa: BLE001  网络/解析等异常
        print(f"    调用失败: {type(exc).__name__}")
        return None

    # 业务侧的"闸门"：不是招聘信息就直接丢弃
    if not job.is_job_post:
        print("    -> 判定为非招聘信息，丢弃")
        return None
    return job


def main() -> None:
    banner("1. 批量抽取：一条坏数据不能炸整批")

    parsed: list[JobPost] = []
    for i, text in enumerate(RAW_POSTS, 1):
        print(f"\n  第 {i} 条: {text[:36]}...")
        job = extract(text)
        if job is None:
            print("    -> 跳过（进入人工复核队列）")
            continue
        parsed.append(job)
        print(f"    -> {job.company} / {job.title} / {job.city} / {job.level.value} / "
              f"{job.salary_min}-{job.salary_max}k / {job.skills}")

    section("2. 校验业务规则：上下限写反会被 Pydantic 拦住")
    try:
        JobPost(
            is_job_post=True, company="X", title="Y", city="Z",
            level=Level.middle, salary_min=30, salary_max=20,
        )
    except ValidationError as exc:
        print(f"  {exc.errors()[0]['msg']}")

    section("3. 结构化结果直接进 DataFrame / 数据库")
    rows = [
        {
            "company": j.company,
            "level": j.level.value,
            "salary_mid": (j.salary_min + j.salary_max) / 2,
            "skills": ",".join(j.skills),
        }
        for j in parsed
    ]
    for row in rows:
        print(f"  {row}")

    section("4. 小结")
    print(f"  成功 {len(parsed)} / {len(RAW_POSTS)} 条")
    print("  生产建议：schema 越具体，模型越不容易自由发挥；")
    print("           校验失败时把错误信息回灌给模型重试一次（ToolStrategy(handle_errors=True) 已内置）。")


if __name__ == "__main__":
    main()
