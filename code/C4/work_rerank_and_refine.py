"""C4-w · 重排 + 精炼（work 版：中文友好分块）

对应文档：docs/chapter4/07_rerank_and_refine.md（C4-07 的练习版本）

和 07 的差别只有**分块策略**：

    07     chunk_size=500 / overlap=100，默认英文分隔符
    work   chunk_size=300 / overlap=20，中文标点优先（。！？；，）

跑完对比一下两边「粗排 vs 精炼后」的条数，就能看出分块粒度的影响。

运行：
    python3 code/C4/work_rerank_and_refine.py
    RAG_LOG=quiet python3 code/C4/work_rerank_and_refine.py
"""
import os as _os, sys as _sys
# 向上找到含 bootstrap.py 的 code/ 目录，加入 sys.path（脚本可能在多级子目录里）
_p = _os.path.dirname(_os.path.abspath(__file__))
while _p != _os.path.dirname(_p) and not _os.path.exists(_os.path.join(_p, "bootstrap.py")):
    _p = _os.path.dirname(_p)
_sys.path.insert(0, _p)
import bootstrap  # noqa: F401  仅为了副作用：加载 code/.env、固定缓存目录、修代理与证书
bootstrap.anchor_cwd(__file__)   # 工作目录钉在脚本所在目录，相对路径才不会失效
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

llm = ChatDeepSeek(
    model="deepseek-chat", 
    temperature=0.1, 
    api_key=os.getenv("DEEPSEEK_API_KEY")
)


# 1. 加载和处理文档
stage = Stage(total=5, title="C4 · 重排 + 精炼（work · 中文分块）")
stage.header("分块用中文标点优先，chunk 更小，看粗排/精炼的差异")

QUERY = "AI还有哪些缺陷需要克服？"
TOPK = 20

with stage.step(
    "加载并切分文档",
    path="../../data/C4/txt/ai.txt",
    chunk="300 / overlap 20（中文标点优先）",
    why="比 07 切得更碎：片段短，LLM 压缩更快，但上下文也更容易被截断",
) as step:
    loader = TextLoader("../../data/C4/txt/ai.txt", encoding="utf-8")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
        chunk_size=300,
        chunk_overlap=20,
    )
    docs = text_splitter.split_documents(documents)
    step.detail(f"原始 {len(documents)} 篇 → 切出 {len(docs)} 个 chunk")
    step.note(f"{len(docs)} chunks")

with stage.step(
    "建 FAISS 向量库 + 基础检索器",
    embed="BAAI/bge-small-zh-v1.5",
    k=TOPK,
) as step:
    vectorstore = FAISS.from_documents(docs, hf_bge_embeddings)
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": TOPK})
    step.note(f"k={TOPK}")

# 3/4. 设置 ColBERT 重排器 + LLM 压缩器
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
    """在 LLMChainExtractor 基础上过滤掉「无」这类占位输出。"""

    def compress_documents(self, documents, query, callbacks=None):
        out = super().compress_documents(documents, query, callbacks=callbacks)
        return [d for d in out if d.page_content.strip() not in ("无", "无。", "NONE", "NO_OUTPUT")]


with stage.step(
    "组装重排 + 压缩管道",
    reranker="ColBERT",
    compressor="LLMChainExtractor（中文提示词）",
) as step:
    reranker = ColBERTReranker()
    compressor = ChineseExtractCompressor.from_llm(llm, prompt=_EXTRACT_PROMPT)
    pipeline_compressor = DocumentCompressorPipeline(
        transformers=[reranker, compressor]
    )
    final_retriever = ContextualCompressionRetriever(
        base_compressor=pipeline_compressor,
        base_retriever=base_retriever,
    )
    step.note("管道就绪")

with stage.step(
    "基础检索（粗排）",
    query=QUERY,
    why="chunk 更碎意味着粗排更容易命中关键词，但也更容易命中「碎片化的噪音」",
) as step:
    base_results = base_retriever.get_relevant_documents(QUERY)
    for i, doc in enumerate(base_results, 1):
        step.detail(f"  [{i}] {doc.page_content[:70]}...")
    step.note(f"Top{len(base_results)}")

with stage.step(
    "重排 + 精炼",
    why="和 07 对比这一行：chunk 越小，精炼后剩下的条数通常越多但每条更短",
) as step:
    final_results = final_retriever.get_relevant_documents(QUERY)
    for i, doc in enumerate(final_results, 1):
        step.detail(f"  [{i}] {' '.join(doc.page_content.split())[:90]}")
    step.detail("")
    step.detail(f"压缩比：{len(base_results)} → {len(final_results)} 条")
    total_chars = sum(len(d.page_content) for d in final_results)
    step.detail(f"精炼后总字符数：{total_chars}")
    step.note(f"{len(base_results)} → {len(final_results)}")

stage.summary()

print()
print("=" * 72)
print("  观察点（对比 07）")
print("=" * 72)
print("1. 07 切 500 字，work 切 300 字：这里是 C4 唯一的变量")
print("2. 中文标点优先（。！？；，）能让一个 chunk 尽量停在句末，语义更完整")
print("3. chunk 小 → 召回多、噪音也多；chunk 大 → 上下文全、token 贵")
print("4. 判断分块好坏没有标准答案，要看下游是「精确问答」还是「摘要归纳」")
print()
print("提示：overlap 从 100 降到 20，是为了减少相邻 chunk 的重复内容")
print("=" * 72)
