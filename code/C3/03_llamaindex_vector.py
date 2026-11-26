"""C3-03 · LlamaIndex：框架化的向量索引

对应文档：docs/chapter3/08_faiss.md（与 02 对照阅读）

这一节的要点：**同样建索引，LangChain 和 LlamaIndex 的分工不一样。**

    LangChain   ：把「嵌入模型 / 向量库 / 文档」当成可替换零件，显式拼起来
    LlamaIndex  ：把「节点解析 → 索引 → 查询」封装成默认流水线，
                  通过 Settings 全局配置模型，写更少的代码

02 的 FAISS 版要显式传 embeddings 给 from_documents；
这里只设置 Settings.embed_model，后续 from_documents 就自动用它。

运行：
    python3 code/C3/03_llamaindex_vector.py
    RAG_LOG=quiet python3 code/C3/03_llamaindex_vector.py

3 个阶段：
    [1/3] 配置全局嵌入模型
    [2/3] 创建索引
    [3/3] 持久化到 llamaindex_index_store/
"""
import os as _os, sys as _sys

# 向上找到含 bootstrap.py 的 code/ 目录，加入 sys.path（脚本可能在多级子目录里）
_p = _os.path.dirname(_os.path.abspath(__file__))
while _p != _os.path.dirname(_p) and not _os.path.exists(_os.path.join(_p, "bootstrap.py")):
    _p = _os.path.dirname(_p)
_sys.path.insert(0, _p)
import bootstrap  # noqa: F401  仅为了副作用：加载 code/.env、固定缓存目录、修代理与证书

bootstrap.anchor_cwd(__file__)  # 工作目录钉在脚本所在目录，相对路径才不会失效
from stage_log import Stage

from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

EMBED_MODEL = "BAAI/bge-small-zh-v1.5"
PERSIST_DIR = "./llamaindex_index_store"

texts = [
    "张三是法外狂徒",
    "LlamaIndex是一个用于构建和查询私有或领域特定数据的框架。",
    "它提供了数据连接、索引和查询接口等工具。",
]

stage = Stage(total=3, title="C3 · LlamaIndex 向量索引")
stage.header("Settings 配模型 → from_documents 建索引 → persist 落盘")

# ---------------------------------------------------------------- 1. 模型
with stage.step(
    "配置全局嵌入模型",
    model=EMBED_MODEL,
    api="Settings.embed_model = HuggingFaceEmbedding(...)",
    why="LlamaIndex 用 Settings 做全局默认，后面不用再显式传模型",
) as step:
    Settings.embed_model = HuggingFaceEmbedding(EMBED_MODEL)
    step.detail("之后所有 from_documents 都会自动使用这个模型")
    step.note("已配置")

# ---------------------------------------------------------------- 2. 建索引
with stage.step(
    "创建索引",
    api="VectorStoreIndex.from_documents(docs)",
    count=len(texts),
) as step:
    docs = [Document(text=t) for t in texts]
    index = VectorStoreIndex.from_documents(docs)
    step.detail(f"{len(docs)} 条文档已切分为节点并编码为向量")
    step.note("已建索引")

# ---------------------------------------------------------------- 3. 持久化
with stage.step(
    "持久化到磁盘",
    api="index.storage_context.persist(persist_dir=...)",
    path=PERSIST_DIR,
) as step:
    index.storage_context.persist(persist_dir=PERSIST_DIR)
    step.detail(f"已写入 {PERSIST_DIR}/（docstore.json + vector_store.json 等）")
    step.note("已落盘")

stage.summary()

print()
print("=" * 72)
print("  观察点：和 02（FAISS 版）对比")
print("=" * 72)
print("1. 02 显式传 embeddings；03 只配一次 Settings，代码更短")
print("2. 02 落盘是 index.faiss + index.pkl 两个文件")
print("   03 落盘是 JSON 一整套（docstore / index_store / vector_store）")
print("3. LlamaIndex 默认还会做 node 切分和元数据维护，为后续检索策略（05/06）铺路")
print("4. 两者底层可以互通：LlamaIndex 也能把 FAISS / Milvus 当作 vector store")
print()
print("提示：本脚本只建索引、不查询；检索见 C3-05 / 06 / 07")
print("=" * 72)
