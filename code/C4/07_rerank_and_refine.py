"""C4-07 · 检索后重排 + 精炼（ColBERT 重排 → LLM 压缩）

对应文档：docs/chapter4/07_rerank_and_refine.md

要点：向量检索拿 Top-K 便宜但粗；**重排（rerank）**用更贵的模型精排；
**精炼（refine / compress）**再把每条文档压成只跟问题相关的几句。

    粗排（向量，快）  → 20 条
    重排（ColBERT，准）→ 重新排序
    精炼（LLM，省）    → 每条只剩相关句子，甚至整条丢掉

运行：
    python3 code/C4/07_rerank_and_refine.py
    RAG_LOG=quiet python3 code/C4/07_rerank_and_refine.py

5 个阶段：
    [1/5] 加载并切分文档
    [2/5] 建 FAISS 向量库 + 基础检索器（k=20）
    [3/5] 组装压缩管道（ColBERT → LLM 抽取）
    [4/5] 基础检索（粗排 Top20）
    [5/5] 重排+精炼后的结果
"""
import os as _os, sys as _sys
# 向上找到含 bootstrap.py 的 code/ 目录，加入 sys.path（脚本可能在多级子目录里）
_p = _os.path.dirname(_os.path.abspath(__file__))
while _p != _os.path.dirname(_p) and not _os.path.exists(_os.path.join(_p, "bootstrap.py")):
    _p = _os.path.dirname(_p)
_sys.path.insert(0, _p)
import bootstrap  # noqa: F401  仅为了副作用：加载 code/.env、修代理与证书
bootstrap.anchor_cwd(__file__)   # 工作目录钉在脚本所在目录，相对路径才不会失效
from llm_config import llm_kwargs
from stage_log import Stage

import os
from langchain_community.vectorstores import FAISS
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_deepseek import ChatDeepSeek

# 导入ColBERT重排器需要的模块
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_core.documents import Document
from typing import Sequence
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

class ColBERTReranker(BaseDocumentCompressor):
    """ColBERT重排器"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        model_name = "bert-base-uncased"

        # 加载模型和分词器
        object.__setattr__(self, 'tokenizer', AutoTokenizer.from_pretrained(model_name))
        object.__setattr__(self, 'model', AutoModel.from_pretrained(model_name))
        self.model.eval()
        print(f"ColBERT模型加载完成")

    def encode_text(self, texts):
        """ColBERT文本编码"""
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        embeddings = outputs.last_hidden_state
        embeddings = F.normalize(embeddings, p=2, dim=-1)

        return embeddings

    def calculate_colbert_similarity(self, query_emb, doc_embs, query_mask, doc_masks):
        """ColBERT相似度计算（MaxSim操作）"""
        scores = []

        for i, doc_emb in enumerate(doc_embs):
            doc_mask = doc_masks[i:i+1]

            # 计算相似度矩阵
            similarity_matrix = torch.matmul(query_emb, doc_emb.unsqueeze(0).transpose(-2, -1))

            # 应用文档mask
            doc_mask_expanded = doc_mask.unsqueeze(1)
            similarity_matrix = similarity_matrix.masked_fill(~doc_mask_expanded.bool(), -1e9)

            # MaxSim操作
            max_sim_per_query_token = similarity_matrix.max(dim=-1)[0]

            # 应用查询mask
            query_mask_expanded = query_mask.unsqueeze(0)
            max_sim_per_query_token = max_sim_per_query_token.masked_fill(~query_mask_expanded.bool(), 0)

            # 求和得到最终分数
            colbert_score = max_sim_per_query_token.sum(dim=-1).item()
            scores.append(colbert_score)

        return scores

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks=None,
    ) -> Sequence[Document]:
        """对文档进行ColBERT重排序"""
        if len(documents) == 0:
            return documents

        # 编码查询
        query_inputs = self.tokenizer(
            [query],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )

        with torch.no_grad():
            query_outputs = self.model(**query_inputs)
            query_embeddings = F.normalize(query_outputs.last_hidden_state, p=2, dim=-1)

        # 编码文档
        doc_texts = [doc.page_content for doc in documents]
        doc_inputs = self.tokenizer(
            doc_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )

        with torch.no_grad():
            doc_outputs = self.model(**doc_inputs)
            doc_embeddings = F.normalize(doc_outputs.last_hidden_state, p=2, dim=-1)

        # 计算ColBERT相似度
        scores = self.calculate_colbert_similarity(
            query_embeddings,
            doc_embeddings,
            query_inputs['attention_mask'],
            doc_inputs['attention_mask']
        )

        # 排序并返回前5个
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        reranked_docs = [doc for doc, _ in scored_docs[:5]]

        return reranked_docs



# 初始化配置
hf_bge_embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-large-zh-v1.5"
)

llm = ChatDeepSeek(temperature=0.1, **llm_kwargs())

stage = Stage(total=5, title="C4 · 重排 + 精炼（ColBERT → LLM 压缩）")
stage.header("先宽召回再精排再压缩：准确率与成本的折中")

QUERY = "AI还有哪些缺陷需要克服？"
TOPK = 20

# 1. 加载和处理文档
with stage.step(
    "加载并切分文档",
    path="../../data/C4/txt/ai.txt",
    chunk="500 / overlap 100",
    why="chunk 大一点能给重排更多上下文，但压缩阶段的 token 成本也会上来",
) as step:
    loader = TextLoader("../../data/C4/txt/ai.txt", encoding="utf-8")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    step.detail(f"原始文档 {len(documents)} 篇 → 切出 {len(docs)} 个 chunk")
    step.note(f"{len(docs)} chunks")

# 2. 创建向量存储和基础检索器
with stage.step(
    "建 FAISS 向量库 + 基础检索器",
    embed="BAAI/bge-small-zh-v1.5",
    k=TOPK,
    why="k 设大一些（20）是为了给重排留足候选 —— 粗排漏掉的，重排也救不回来",
) as step:
    vectorstore = FAISS.from_documents(docs, hf_bge_embeddings)
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": TOPK})
    step.note(f"k={TOPK}")

# 3. 设置ColBERT重排序器 + LLM压缩器
# 注意：LLMChainExtractor 自带的提示词是英文的（"...extract any part of the context AS IS..."），
# 中文场景下小模型（如本地 3B）经常直接返回空串，导致压缩后一条不剩。
# 这里换成中文提示词，并要求「无相关内容」时输出固定词，便于过滤。
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

_EXTRACT_PROMPT = PromptTemplate.from_template(
    """下面是一段文档摘录。请从中摘抄出与问题相关的句子，**原样照抄、不要改写**。
如果整段都不相关，只输出：无

问题：{question}
摘录：
{context}

摘抄结果："""
)

class ChineseExtractCompressor(LLMChainExtractor):
    """在 LLMChainExtractor 基础上过滤掉「无」这类占位输出。

    基类只判断 len(output) == 0，而中文提示词让模型输出「无」时长度不为 0，
    于是「无相关」的文档也会被当成结果返回。
    """

    def compress_documents(self, documents, query, callbacks=None):
        out = super().compress_documents(documents, query, callbacks=callbacks)
        return [d for d in out if d.page_content.strip() not in ("无", "无。", "NONE", "NO_OUTPUT")]



with stage.step(
    "组装重排 + 压缩管道",
    reranker="ColBERT",
    compressor="LLMChainExtractor（中文提示词）",
    why="ColBERT 逐 token 交互打分，比单向量点积准，但只能重排不能召回",
) as step:
    reranker = ColBERTReranker()
    compressor = ChineseExtractCompressor.from_llm(llm, prompt=_EXTRACT_PROMPT)

    # 5. 组装压缩管道：ColBERT 重排 -> LLM 压缩
    pipeline_compressor = DocumentCompressorPipeline(
        transformers=[reranker, compressor]
    )

    # 6. 创建最终的压缩检索器
    final_retriever = ContextualCompressionRetriever(
        base_compressor=pipeline_compressor,
        base_retriever=base_retriever,
    )
    step.note("管道就绪")

# 7. 执行查询并展示结果
with stage.step(
    "基础检索（粗排）",
    query=QUERY,
    why="这是「没有重排」的基线：Top-K 里通常混着几条不相关的",
) as step:
    base_results = base_retriever.get_relevant_documents(QUERY)
    for i, doc in enumerate(base_results, 1):
        step.detail(f"  [{i}] {doc.page_content[:80]}...")
    step.note(f"Top{len(base_results)}")

with stage.step(
    "重排 + 精炼",
    why="条数通常会明显变少：不相关的被 LLM 判成「无」后直接丢弃，token 降下来了",
) as step:
    final_results = final_retriever.get_relevant_documents(QUERY)
    for i, doc in enumerate(final_results, 1):
        step.detail(f"  [{i}] {' '.join(doc.page_content.split())[:100]}")
    step.note(f"{len(base_results)} → {len(final_results)} 条")

stage.summary()

print()
print("=" * 72)
print("  观察点")
print("=" * 72)
print(f"1. 粗排 {len(base_results)} 条 → 精炼后 {len(final_results)} 条，数量差就是「噪音」")
print("2. ColBERT 只能重排不能召回：它要给每对 (query, doc) 算交互，全库排序太贵")
print("3. LLM 压缩的收益是「省 token」，代价是每条文档一次 LLM 调用 —— 20 条就是 20 次")
print("4. 生产里常把压缩换成更便宜的模型，或者只对 Top-5 压缩")
print()
print("提示：LLMChainExtractor 自带英文提示词，中文场景常返回空串，所以这里换了中文 prompt")
print("=" * 72)
