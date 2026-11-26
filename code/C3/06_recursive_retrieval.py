"""C3-06 · 递归检索：先找「去哪个表查」，再让那张表自己算

对应文档：docs/chapter3/06_recursive_retrieval.md

这一节的要点：**有些问题不适合用「向量相似度 + 文本片段」回答，而适合「查表」。**

    movie.xlsx 里每个 sheet 是一个年份的电影表。
    问题「1994 年评分人数最少的电影」需要：找到 1994 的 sheet → 在那张表上做聚合。

递归检索干的事：
    第一层：在「摘要节点」里检索，命中 → 年份_1994
    第二层：该节点的 index_id 指向一个 PandasQueryEngine，
            把问题交给它，让它生成 pandas 代码算结果

    RecursiveRetriever 的 retriever_dict / query_engine_dict 就是这两层的接线表。

运行（要 LLM，约 30-60 秒）：
    python3 code/C3/06_recursive_retrieval.py
    RAG_LOG=quiet python3 code/C3/06_recursive_retrieval.py

5 个阶段：
    [1/5] 配置模型
    [2/5] 读 Excel，每个 sheet 建一个 Pandas 查询引擎 + 一个摘要节点
    [3/5] 用摘要节点建顶层向量索引
    [4/5] 组装 RecursiveRetriever
    [5/5] 查询并观察两跳过程
"""
import os as _os, sys as _sys

# 向上找到含 bootstrap.py 的 code/ 目录，加入 sys.path（脚本可能在多级子目录里）
_p = _os.path.dirname(_os.path.abspath(__file__))
while _p != _os.path.dirname(_p) and not _os.path.exists(_os.path.join(_p, "bootstrap.py")):
    _p = _os.path.dirname(_p)
_sys.path.insert(0, _p)
import bootstrap  # noqa: F401  仅为了副作用：加载 code/.env、修代理与证书

bootstrap.anchor_cwd(__file__)  # 工作目录钉在脚本所在目录，相对路径才不会失效
from compat_shims import install as _install_compat_shims

_install_compat_shims()  # 绕开 mistralai 3.x 导致的 llama_index.experimental 导入失败

from llm_config import openai_llm_kwargs
from stage_log import Stage

import pandas as pd
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core.schema import IndexNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.experimental.query_engine import PandasQueryEngine
from llama_index.llms.deepseek import DeepSeek

EXCEL_FILE = "../../data/C3/excel/movie.xlsx"
EMBED_MODEL = "BAAI/bge-small-zh-v1.5"
QUERY = "1994年评分人数最少的电影是哪一部？"

stage = Stage(total=5, title="C3 · 递归检索（Excel 多表）")
stage.header("摘要索引路由 → 命中工作表 → Pandas 引擎计算")

# ---------------------------------------------------------------- 1. 模型
with stage.step(
    "配置模型",
    llm=openai_llm_kwargs().get("model"),
    embed=EMBED_MODEL,
) as step:
    Settings.llm = DeepSeek(**openai_llm_kwargs())
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
    step.note("已配置")

# ---------------------------------------------------------------- 2. 建子引擎
with stage.step(
    "为每个工作表建查询引擎 + 摘要节点",
    file=EXCEL_FILE,
    why="每个 sheet 一个 PandasQueryEngine；摘要节点是它在顶层索引里的「门牌号」",
) as step:
    xls = pd.ExcelFile(EXCEL_FILE)
    df_query_engines = {}
    all_nodes = []

    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name)

        query_engine = PandasQueryEngine(df=df, llm=Settings.llm, verbose=True)

        year = sheet_name.replace("年份_", "")
        summary = f"这个表格包含了年份为 {year} 的电影信息，可以用来回答关于这一年电影的具体问题。"
        # index_id 就是接线用的 key：递归检索靠它把摘要节点和子引擎对上
        node = IndexNode(text=summary, index_id=sheet_name)
        all_nodes.append(node)

        df_query_engines[sheet_name] = query_engine
        step.detail(f"{sheet_name:<12} {len(df):>3} 行  → 摘要节点已建")

    step.note(f"{len(df_query_engines)} 个子引擎")

# ---------------------------------------------------------------- 3. 顶层索引
with stage.step(
    "用摘要节点建顶层向量索引",
    api="VectorStoreIndex(all_nodes)",
    why="顶层只索引摘要，不索引数据本身 —— 这就是「递归」的意思",
) as step:
    vector_index = VectorStoreIndex(all_nodes)
    step.detail(f"{len(all_nodes)} 个摘要节点已编码（每个只描述一个年份）")
    step.note(f"{len(all_nodes)} 节点")

# ---------------------------------------------------------------- 4. 递归检索器
with stage.step(
    "组装 RecursiveRetriever",
    retriever_dict="{'vector': 顶层向量检索器}",
    query_engine_dict="{'年份_xxxx': Pandas 引擎}",
    why="两本字典合起来定义「命中哪一层就往哪一跳」",
) as step:
    vector_retriever = vector_index.as_retriever(similarity_top_k=1)
    recursive_retriever = RecursiveRetriever(
        "vector",
        retriever_dict={"vector": vector_retriever},
        query_engine_dict=df_query_engines,
        verbose=True,
    )
    query_engine = RetrieverQueryEngine.from_args(recursive_retriever)
    step.detail("第一跳：vector（找摘要节点）")
    step.detail("第二跳：index_id → 对应的 PandasQueryEngine")
    step.note("已组装")

# ---------------------------------------------------------------- 5. 查询
with stage.step(
    "执行查询",
    query=QUERY,
    why="下面的 Retrieving with query id 就是两跳的痕迹，注意看它跳到了哪个年份",
) as step:
    response = query_engine.query(QUERY)
    step.detail(f"最终回答：{response}")
    step.note(f"{str(response)[:20]}")

stage.summary()

print()
print("=" * 72)
print("  观察点")
print("=" * 72)
print("1. 日志里会出现两行 Retrieving：先 query id None（顶层），再跳到 年份_1994")
print("2. 生成的是 pandas 代码，不是文本检索：")
print("     df[df['评分人数'] == df['评分人数'].min()]['电影名称'].iloc[0]")
print("3. 这类问题用纯向量检索答不出来 —— 相似度找得到「相关段落」，算不出「最小值」")
print("4. 关键设计：index_id 必须和 query_engine_dict 的 key 完全一致，否则跳不过去")
print()
print("提示：PandasQueryEngine 会把问题转成代码执行，务必只对可信数据启用")
print("=" * 72)
