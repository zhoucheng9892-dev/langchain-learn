"""第 10 章 · 公共模块：离线可跑的确定性 Embedding

真实项目请换成：
    from langchain_openai import OpenAIEmbeddings      # 或
    from langchain_huggingface import HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
示例里用自实现的哈希向量，是为了：
    1. 不需要额外的 embedding API（DeepSeek 目前不提供 embeddings 接口）；
    2. 结果完全确定，每次运行输出一致，方便对照文档；
    3. 把 RAG 的"流程"和"向量模型"解耦，看清每一步在做什么。

原理：把文本切成字符二元组（bigram），每个 bigram 哈希到固定维度的某一维，
再整体做 L2 归一化。语义能力远不如真实模型，但关键词重合度高的文本相似度会明显更高，
足以演示检索流程。
"""

from __future__ import annotations

import hashlib
import math
import re

from langchain_core.embeddings import Embeddings


class HashingEmbeddings(Embeddings):
    """字符 bigram 哈希向量。仅用于教学与测试。"""

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        cleaned = re.sub(r"\s+", "", text.lower())
        vec = [0.0] * self.dim
        for i in range(max(len(cleaned) - 1, 1)):
            gram = cleaned[i : i + 2]
            digest = hashlib.md5(gram.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)
