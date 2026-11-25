"""C1 · 基于 LlamaIndex 的四步 RAG（带分阶段日志）

对应文档：docs/chapter1/03_get_start_rag.md 第四节

LlamaIndex 把「分片 → 向量化 → 写入向量库」全藏进了 `VectorStoreIndex.from_documents()`，
一句代码就把 LangChain 的 3 步干完了。所以这里的日志会额外标注：
**每一句 API 内部到底替你做了哪几件事**。

6 个阶段：
    [1/6] 初始化 LLM 与嵌入模型
    [2/6] 加载文档
    [3/6] 构建索引（内部：分片 → 向量化 → 写入向量库）
    [4/6] 创建查询引擎
    [5/6] 相似检索（用 as_retriever() 把隐藏的检索步骤显式拉出来看）
    [6/6] LLM 生成回答

想关掉这些日志：RAG_LOG=quiet python3 02_llamaIndex_example.py
"""

import os as _os, sys as _sys

# 向上找到含 bootstrap.py 的 code/ 目录，加入 sys.path（脚本可能在多级子目录里）
_p = _os.path.dirname(_os.path.abspath(__file__))
while _p != _os.path.dirname(_p) and not _os.path.exists(
    _os.path.join(_p, "bootstrap.py")
):
    _p = _os.path.dirname(_p)
_sys.path.insert(0, _p)
import bootstrap  # noqa: F401  仅为了副作用：加载 code/.env、修代理与证书

bootstrap.anchor_cwd(__file__)  # 工作目录钉在脚本所在目录，相对路径才不会失效
from llm_config import openai_llm_kwargs
from stage_log import Stage

import os

# os.environ['HF_ENDPOINT']='https://hf-mirror.com'
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

load_dotenv()

stage = Stage(total=6, title="C1 · RAG 流水线")
stage.header("LlamaIndex 版 · 低代码，四步被框架封装")

MD_PATH = ["../../data/C1/markdown/easy-rl-chapter1.md"]
QUESTION = "文中举了哪些例子?"

# ---------------------------------------------------------------- 1. 初始化
with stage.step(
    "初始化 LLM 与嵌入模型",
    llm=openai_llm_kwargs()["model"],
    endpoint=openai_llm_kwargs()["api_base"],
    embed_model="BAAI/bge-small-zh-v1.5",
) as step:
    # 默认使用 AIHubmix；配置了 LLM_BASE_URL（例如本地 Ollama）时优先走本地/自建端点
    Settings.llm = OpenAILike(
        is_chat_model=True,
        context_window=32768,
        **openai_llm_kwargs(),
    )
    Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-small-zh-v1.5")
    step.detail("Settings 是全局配置，后面所有索引/查询引擎都默认用它俩")
    step.note("两个模型就绪")

# ---------------------------------------------------------------- 2. 加载文档
with stage.step(
    "加载文档",
    reader="SimpleDirectoryReader",
    files=str(MD_PATH),
) as step:
    documents = SimpleDirectoryReader(input_files=MD_PATH).load_data()
    step.detail(
        f"{len(documents)} 个 Document，"
        f"共 {sum(len(d.text) for d in documents)} 字符"
    )
    step.note("等价于 LangChain 的 UnstructuredMarkdownLoader")

# ---------------------------------------------------------------- 3. 构建索引
with stage.step(
    "构建索引",
    api="VectorStoreIndex.from_documents(documents)",
    note="这一句内部做了 3 件事：分片 → 向量化 → 写入向量库",
) as step:
    index = VectorStoreIndex.from_documents(documents)

    # 把内部结果挖出来看看，证明它确实替我们做了分片和向量化
    nodes = index.docstore.docs

    # 向量存在 SimpleVectorStore 的 embedding_dict 里：{node_id: [浮点数...]}
    emb_dict = index.vector_store.data.embedding_dict
    dim = len(next(iter(emb_dict.values()))) if emb_dict else 0

    lens = [len(n.text) for n in nodes.values()]
    step.detail(
        f"分片：共 {len(nodes)} 个 node（默认 SentenceSplitter(chunk_size=1024, overlap=200)）"
    )
    step.detail(f"      每块 {min(lens)}~{max(lens)} 字符，均 {sum(lens) // len(lens)}")
    step.detail("      ⚠ chunk_size 的单位是 **token** 不是字符（LangChain 按字符算）")
    step.detail(f"向量化：{len(emb_dict)} 条 × {dim} 维")
    step.detail(f"写入：{type(index.vector_store).__name__}（存在内存里）")
    step.note(f"{len(nodes)} 个 node，1 句话干完 LangChain 的 3 步")

# ---------------------------------------------------------------- 4. 查询引擎
with stage.step(
    "创建查询引擎",
    api="index.as_query_engine(similarity_top_k=3)",
    note="把「检索器 + 提示词 + LLM」打包成一个对象",
) as step:
    # 显式写 similarity_top_k=3：默认是 2，不写会和上面第 5 步看到的检索结果对不上
    query_engine = index.as_query_engine(similarity_top_k=3)
    step.detail(
        "内部 = VectorIndexRetriever(top_k=3) + ResponseSynthesizer(提示词模板 + LLM)"
    )
    step.detail("不传参数时默认 top_k=2，很多『怎么只回了 2 条』的疑惑都来自这里")
    step.note("引擎就绪")

# ---------------------------------------------------------------- 5. 相似检索
with stage.step(
    "llm 相似检索",
    question=QUESTION,
    api="index.as_retriever(similarity_top_k=3)",
    note="LlamaIndex 的 query() 会自己检索，这里拆出来单独看",
) as step:
    retriever = index.as_retriever(similarity_top_k=3)
    nodes = retriever.retrieve(QUESTION)
    step.note(f"nodes: {nodes}")
    for i, n in enumerate(nodes, 1):
        head = n.text[:56].replace("\n", " ")
        step.detail(f"top{i} score={n.score:.4f}  {head}…")
    step.note(
        f"命中 {len(nodes)} 个 node，"
        f"拼成上下文 {sum(len(n.text) for n in nodes)} 字符"
    )

# ---------------------------------------------------------------- 6. 生成
with stage.step(
    "llm 生成回答",
    api="query_engine.query(question)",
    note="这一步会把上面检索到的 node 塞进提示词再调 LLM",
) as step:
    step.detail("提示词模板（换行已压平）：")
    for name, p in query_engine.get_prompts().items():
        text = getattr(p, "default_template", p)
        text = getattr(text, "template", str(text))
        flat = " ".join(str(text).split())
        short = name.replace("response_synthesizer:", "")
        step.detail(f"  {short}: {flat[:88]}…")

    response = query_engine.query(QUESTION)
    step.note(f"本次实际用了 {len(response.source_nodes)} 个 node 作为上下文")
    for i, sn in enumerate(response.source_nodes, 1):
        step.detail(f"  [{i}] score={sn.score:.4f}  {sn.node.text[:44]!r}…")
    step.note("生成完成")

stage.summary()

print()
print("=" * 72)
print("  最终回答（response）")
print("=" * 72)
print(response)
print("=" * 72)
print(response.source_nodes)
print("=" * 72)
print(query_engine.get_prompts().items())
print("=" * 72)
print("说明：LlamaIndex 返回的是 Response 对象，直接 print 就是答案文本；")
print("      想看检索到的原文用 response.source_nodes。")
