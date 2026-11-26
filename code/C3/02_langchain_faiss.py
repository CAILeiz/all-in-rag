"""C3-02 · LangChain + FAISS：最小的本地向量库

对应文档：docs/chapter3/08_faiss.md

这一节的要点：**向量库就是「向量 + 原文 + 检索算法」的组合。**
FAISS 负责高效的相似度搜索，LangChain 负责把文档、嵌入模型、
向量库串成一条流水线。

流程只有四步，但每一步都可以单独替换：
    文档 → 嵌入模型 → 向量库（落盘）→ 相似度检索

运行：
    python3 code/C3/02_langchain_faiss.py
    RAG_LOG=quiet python3 code/C3/02_langchain_faiss.py

4 个阶段：
    [1/4] 准备文档
    [2/4] 加载嵌入模型（bge-small-zh，首次约 10 秒）
    [3/4] 建索引并落盘到 faiss_index_store/
    [4/4] 从磁盘重新加载后检索
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

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

EMBED_MODEL = "BAAI/bge-small-zh-v1.5"
INDEX_DIR = "./faiss_index_store"

texts = [
    "张三是法外狂徒",
    "FAISS是一个用于高效相似性搜索和密集向量聚类的库。",
    "LangChain是一个用于开发由语言模型驱动的应用程序的框架。",
]
QUERY = "FAISS是做什么的？"

stage = Stage(total=4, title="C3 · FAISS 向量库")
stage.header("LangChain 串起 文档 → 嵌入 → FAISS → 检索")

# ---------------------------------------------------------------- 1. 文档
with stage.step("准备文档", count=len(texts), note="演示用，真实场景由 C2 的分块结果提供") as step:
    docs = [Document(page_content=t) for t in texts]
    for i, d in enumerate(docs, 1):
        step.detail(f"[{i}] {d.page_content}")
    step.note(f"{len(docs)} 条")

# ---------------------------------------------------------------- 2. 嵌入
with stage.step(
    "加载嵌入模型",
    model=EMBED_MODEL,
    why="FAISS 只存向量，把文字变成向量的是这个模型",
) as step:
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    step.detail("模型就绪：中文小模型，512 维，CPU 即可")
    step.note("已加载")

# ---------------------------------------------------------------- 3. 建索引
with stage.step(
    "建索引并落盘",
    api="FAISS.from_documents(docs, embeddings).save_local(...)",
    path=INDEX_DIR,
) as step:
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(INDEX_DIR)
    step.detail(f"索引已写入 {INDEX_DIR}/（index.faiss + index.pkl）")
    step.note("已落盘")

# ---------------------------------------------------------------- 4. 检索
with stage.step(
    "加载并检索",
    api="FAISS.load_local(...).similarity_search(query, k=1)",
    query=QUERY,
    why="重新从磁盘加载，验证索引真的持久化了，而不是只存在内存里",
) as step:
    # allow_dangerous_deserialization：加载 pkl 会执行反序列化，
    # 只应加载自己生成的索引，别加载来路不明的文件
    loaded = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    results = loaded.similarity_search(QUERY, k=1)
    for doc in results:
        step.detail(f"命中：{doc.page_content}")
    step.note(f"Top1 命中")

stage.summary()

print()
print("=" * 72)
print("  观察点")
print("=" * 72)
print("1. 查询「FAISS是做什么的？」命中了讲 FAISS 的那句 —— 语义匹配，不是关键词匹配")
print("2. 索引落盘后可用 load_local 重新加载，进程退出不丢数据")
print("3. FAISS 是「本地文件」型向量库：没有服务端，适合单机和小数据量")
print("4. 数据量上万、要并发、要增删改 → 换 C3-04 的 Milvus（服务端型）")
print()
print("提示：allow_dangerous_deserialization=True 只对本机生成的索引开启")
print("=" * 72)
