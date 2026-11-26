"""C3-05 · 句子窗口检索：小块检索、大块喂给模型

对应文档：docs/chapter3/05_advanced_rag_retrieval.md

这一节的要点：**「检索用的块」和「给模型的块」不必是同一个块。**

    句子窗口检索的做法：
        1. 切成一句一句（句子很小，检索最准）
        2. 每个句子记下它前后各 3 句，存进 metadata["window"]
        3. 命中某个句子后，把它的 window（7 句）交给 LLM

    MetadataReplacementPostProcessor 就是干第 3 步的：
    把节点的正文替换成 window 里的内容。

对比基准：常规 SentenceSplitter（512 字符）直接检索。

运行（要 PDF + LLM，约 1-3 分钟）：
    python3 code/C3/05_sentence_window_retrieval.py
    RAG_LOG=quiet python3 code/C3/05_sentence_window_retrieval.py

5 个阶段：
    [1/5] 配置 LLM 与嵌入模型
    [2/5] 加载 PDF
    [3/5] 构建两个索引（句子窗口 / 常规）
    [4/5] 组装查询引擎（带 window 替换）
    [5/5] 同一问题，两种检索对比
"""
import os as _os, sys as _sys

# 向上找到含 bootstrap.py 的 code/ 目录，加入 sys.path（脚本可能在多级子目录里）
_p = _os.path.dirname(_os.path.abspath(__file__))
while _p != _os.path.dirname(_p) and not _os.path.exists(_os.path.join(_p, "bootstrap.py")):
    _p = _os.path.dirname(_p)
_sys.path.insert(0, _p)
import bootstrap  # noqa: F401  仅为了副作用：加载 code/.env、修代理与证书

bootstrap.anchor_cwd(__file__)  # 工作目录钉在脚本所在目录，相对路径才不会失效
from llm_config import openai_llm_kwargs
from stage_log import Stage

from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter, SentenceWindowNodeParser
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.deepseek import DeepSeek

PDF_PATH = "../../data/C3/pdf/IPCC_AR6_WGII_Chapter03.pdf"
EMBED_MODEL = "BAAI/bge-small-en"
QUERY = "What are the concerns surrounding the AMOC?"

stage = Stage(total=5, title="C3 · 句子窗口检索")
stage.header("小块检索 → 用相邻句子当上下文 → LLM 回答")

# ---------------------------------------------------------------- 1. 模型
with stage.step(
    "配置 LLM 与嵌入模型",
    llm=openai_llm_kwargs().get("model"),
    embed=EMBED_MODEL,
    why="嵌入模型决定检索质量，LLM 只负责最后的组织语言",
) as step:
    Settings.llm = DeepSeek(temperature=0.1, **openai_llm_kwargs())
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
    step.detail(f"LLM 端点：{openai_llm_kwargs().get('api_base')}")
    step.note("已配置")

# ---------------------------------------------------------------- 2. 加载
with stage.step(
    "加载 PDF",
    loader="SimpleDirectoryReader",
    file=PDF_PATH,
) as step:
    documents = SimpleDirectoryReader(input_files=[PDF_PATH]).load_data()
    step.detail(f"{len(documents)} 个 Document，共 {sum(len(d.text) for d in documents)} 字符")
    step.note(f"{len(documents)} 页")

# ---------------------------------------------------------------- 3. 索引
with stage.step(
    "构建两个索引",
    window_size=3,
    base_chunk=512,
    why="两个索引吃同一份文档，只有切法不同，这样对比才公平",
) as step:
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=3,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )
    sentence_nodes = node_parser.get_nodes_from_documents(documents)
    sentence_index = VectorStoreIndex(sentence_nodes)

    base_parser = SentenceSplitter(chunk_size=512)
    base_nodes = base_parser.get_nodes_from_documents(documents)
    base_index = VectorStoreIndex(base_nodes)

    step.detail(f"句子窗口：{len(sentence_nodes)} 个节点（每个节点 = 1 句 + 前后窗口）")
    step.detail(f"常规分块：{len(base_nodes)} 个节点（每块约 512 字符）")
    step.note(f"{len(sentence_nodes)} / {len(base_nodes)} 节点")

# ---------------------------------------------------------------- 4. 查询引擎
with stage.step(
    "组装查询引擎",
    postprocessor="MetadataReplacementPostProcessor",
    top_k=2,
    why="这一步把「命中的句子」换成「它周围的窗口」，是本节的核心机关",
) as step:
    sentence_query_engine = sentence_index.as_query_engine(
        similarity_top_k=2,
        node_postprocessors=[
            MetadataReplacementPostProcessor(target_metadata_key="window")
        ],
    )
    base_query_engine = base_index.as_query_engine(similarity_top_k=2)
    step.detail("窗口引擎：检索 2 个句子 → 各自展开成 7 句 → 送 LLM")
    step.detail("基准引擎：直接检索 2 个 512 字符块 → 送 LLM")
    step.note("2 个引擎")

# ---------------------------------------------------------------- 5. 对比
with stage.step(
    "两种检索对比",
    query=QUERY,
    why="同一问题、同一份文档，只换检索策略，看回答差多少",
) as step:
    step.detail("--- 句子窗口检索结果 ---")
    window_response = sentence_query_engine.query(QUERY)
    for line in str(window_response).strip().splitlines()[:6]:
        step.detail(f"  {line}")

    step.detail("")
    step.detail("--- 常规检索结果 ---")
    base_response = base_query_engine.query(QUERY)
    for line in str(base_response).strip().splitlines()[:6]:
        step.detail(f"  {line}")
    step.note("2 组回答")

stage.summary()

print()
print("=" * 72)
print("  观察点")
print("=" * 72)
print("1. 检索单元是「一句话」，所以定位很准；但一句话不够回答，所以要窗口")
print("2. 窗口大小 window_size=3 是权衡：越大上下文越全，也越容易引入无关内容")
print("3. 代价：节点数变多（每句一个），索引体积和检索开销都上升")
print("4. 常规分块的做法是「块切小但留 overlap」，句子窗口是「块切极小、上下文外挂」，")
print("   后者更灵活：同一份句子可以配不同大小的窗口，不用重建向量")
print()
print("提示：回答依赖本地 LLM，端点由 code/.env 的 LLM_BASE_URL 决定")
print("=" * 72)
