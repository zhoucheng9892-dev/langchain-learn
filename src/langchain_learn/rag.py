"""RAG 公共构建函数：加载 -> 切分 -> 向量化 -> 检索器。

第 10 章的几个示例共用这一份实现，避免每个文件重复 30 行。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from .embeddings import HashingEmbeddings

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def load_documents() -> list[Document]:
    """读取 data/ 下的 markdown。"""
    docs: list[Document] = []
    for path in sorted(DATA_DIR.glob("*.md")):
        docs.append(Document(page_content=path.read_text(encoding="utf-8"), metadata={"source": path.name}))
    return docs


def split_documents(docs: list[Document], chunk_size: int = 400, overlap: int = 60) -> list[Document]:
    """按 Markdown 标题 + 长度两级切分。"""
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2")], strip_headers=False
    )
    length_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap, separators=["\n\n", "\n", "。", "，", " "]
    )
    chunks: list[Document] = []
    for doc in docs:
        for section_doc in header_splitter.split_text(doc.page_content):
            for piece in length_splitter.split_text(section_doc.page_content):
                chunks.append(
                    Document(
                        page_content=piece,
                        metadata={
                            "source": doc.metadata["source"],
                            "headers": " > ".join(section_doc.metadata.values()) or "-",
                        },
                    )
                )
    return chunks


@lru_cache(maxsize=1)
def build_vector_store() -> InMemoryVectorStore:
    """构建（并缓存）内存向量库。"""
    chunks = split_documents(load_documents())
    store = InMemoryVectorStore(embedding=HashingEmbeddings(dim=512))
    store.add_documents(chunks)
    return store


def build_retriever(k: int = 3):
    """返回一个可 invoke 的检索器。"""
    return build_vector_store().as_retriever(search_kwargs={"k": k})
