"""第 10 章 · 示例 1：索引与检索（RAG 的前半段）

运行：
    uv run python chapters/ch10_rag/01_index_and_retrieve.py

RAG 三步：切分 -> 向量化 -> 存储检索。
顺序永远是：Load（加载）-> Split（切分）-> Embed（向量化）-> Store（存储）-> Retrieve（检索）。

注意切分策略比模型更影响效果：
    切太大 -> 检索精准度下降，一次塞进太多无关内容；
    切太小 -> 语义被切碎，检索到的是"半句话"。
中文场景常用 chunk_size=300~500、overlap=50~100，按标题/段落优先切。
"""

from pathlib import Path

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from langchain_learn import banner, section
from langchain_learn.embeddings import HashingEmbeddings

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def load_documents() -> list[dict]:
    """加载 data/ 下的 markdown，返回 (来源文件, 内容) 列表。"""
    docs = []
    for path in sorted(DATA_DIR.glob("*.md")):
        docs.append({"source": path.name, "text": path.read_text(encoding="utf-8")})
    return docs


def main() -> None:
    banner("1. 加载：3 篇文档，共多少字")
    raw_docs = load_documents()
    total_chars = 0
    for d in raw_docs:
        print(f"  {d['source']:<24} {len(d['text']):>5} 字")
        total_chars += len(d["text"])
    print(f"  合计 {total_chars} 字")

    section("2. 切分：先按 Markdown 标题切，再按长度兜底")
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2")],
        strip_headers=False,  # 保留标题，检索时标题本身就是强信号
    )
    length_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400, chunk_overlap=60, separators=["\n\n", "\n", "。", "，", " "]
    )

    chunks = []
    for doc in raw_docs:
        for section_doc in header_splitter.split_text(doc["text"]):
            # 太长的二级标题段落再按长度切
            for piece in length_splitter.split_text(section_doc.page_content):
                chunks.append(
                    {
                        "text": piece,
                        "source": doc["source"],
                        "headers": " > ".join(section_doc.metadata.values()) or "-",
                    }
                )

    print(f"  切出 {len(chunks)} 个片段，平均 {sum(len(c['text']) for c in chunks) // len(chunks)} 字")
    for c in chunks[:3]:
        print(f"    [{c['source']} | {c['headers']}] {c['text'][:40]}...")

    section("3. 向量化 + 入库")
    embeddings = HashingEmbeddings(dim=512)
    vector_store = InMemoryVectorStore(embedding=embeddings)
    vector_store.add_texts(
        texts=[c["text"] for c in chunks],
        metadatas=[{"source": c["source"], "headers": c["headers"]} for c in chunks],
    )
    print(f"  已写入 {len(chunks)} 条向量，维度 512")
    print("  生产替换：InMemoryVectorStore -> Chroma / PGVector / Milvus / Qdrant")

    section("4. 检索：相似度搜索 top-3")
    query = "中间件有哪些钩子？before_model 和 after_model 分别做什么？"
    print(f"  查询: {query}")
    results = vector_store.similarity_search_with_score(query, k=3)
    for i, (doc, score) in enumerate(results, 1):
        print(f"\n  [{i}] score={score:.3f} 来源={doc.metadata['source']} 小节={doc.metadata['headers']}")
        print(f"      {doc.page_content[:110].replace(chr(10), ' ')}...")

    section("5. 换成 MMR：既相关又不重复")
    retriever = vector_store.as_retriever(
        search_type="mmr", search_kwargs={"k": 2, "fetch_k": 6, "lambda_mult": 0.5}
    )
    for doc in retriever.invoke(query):
        print(f"  - [{doc.metadata['source']}] {doc.page_content[:60]}...")

    section("6. 检索器再包一层：过滤 + 阈值")
    filtered = vector_store.similarity_search_with_score(query, k=5)
    good = [(d, s) for d, s in filtered if s >= 0.05]  # 阈值检索：过滤掉明显不相关的
    print(f"  top-5 中高于阈值的有 {len(good)} 条")
    print("  说明：真实向量模型的相关性分数分布不同，阈值必须用你自己的数据标定。")


if __name__ == "__main__":
    main()
