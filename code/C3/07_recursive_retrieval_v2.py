"""C3-07 · 递归检索 v2：用「元数据过滤」代替「子引擎」

对应文档：docs/chapter3/06_recursive_retrieval.md（与 06 对照阅读）

这一节的要点：**同样是两层路由，但第二层不调 Pandas 引擎，而是加过滤条件再检索。**

    06（RecursiveRetriever）              07（v2：元数据过滤）
    ----------------------------------    ---------------------------------------------------
    顶层检索摘要节点                       顶层检索摘要文档
    命中的 index_id → PandasQueryEngine    命中的 metadata["sheet_name"] → 作为过滤条件
    第二层「执行代码」算出结果              第二层「在内容索引里按 sheet_name 过滤」再检索

07 的好处：不执行模型生成的代码，安全性和可预测性更好；
代价：它只做「检索」，复杂聚合（求最小值）仍依赖 LLM 读表格后自己算，
      所以对算术类问题不如 06 稳。

运行（要 LLM，约 30-60 秒）：
    python3 code/C3/07_recursive_retrieval_v2.py
    RAG_LOG=quiet python3 code/C3/07_recursive_retrieval_v2.py

6 个阶段：
    [1/6] 配置模型
    [2/6] 读 Excel、清洗数据、生成摘要/内容两类文档
    [3/6] 建摘要索引
    [4/6] 建内容索引
    [5/6] 两步查询：先路由，再过滤检索
    [6/6] 汇总回答
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

import pandas as pd
from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.deepseek import DeepSeek

EXCEL_FILE = "../../data/C3/excel/movie.xlsx"
EMBED_MODEL = "BAAI/bge-small-zh-v1.5"
QUERY = "1994年评分人数最少的电影是哪一部？"

stage = Stage(total=6, title="C3 · 递归检索 v2（元数据过滤）")
stage.header("摘要索引路由 → 取 sheet_name → 内容索引按元数据过滤再检索")

# ---------------------------------------------------------------- 1. 模型
with stage.step("配置模型", llm=openai_llm_kwargs().get("model"), embed=EMBED_MODEL) as step:
    Settings.llm = DeepSeek(**openai_llm_kwargs())
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
    step.note("已配置")

# ---------------------------------------------------------------- 2. 数据
with stage.step(
    "加载 Excel 并生成两类文档",
    file=EXCEL_FILE,
    why="摘要文档给「路由」用（短、只描述年份），内容文档给「回答」用（整张表）",
) as step:
    xls = pd.ExcelFile(EXCEL_FILE)
    summary_docs = []
    content_docs = []

    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name)

        # 评分人数形如「12345人评价」，清成纯数字，否则没法比较大小
        if "评分人数" in df.columns:
            df["评分人数"] = df["评分人数"].astype(str).str.replace("人评价", "").str.strip()
            df["评分人数"] = pd.to_numeric(df["评分人数"], errors="coerce").fillna(0).astype(int)

        year = sheet_name.replace("年份_", "")
        summary_docs.append(
            Document(
                text=f"这个表格包含了年份为 {year} 的电影信息，包括电影名称、导演、评分、评分人数等。",
                metadata={"sheet_name": sheet_name},
            )
        )
        content_docs.append(
            Document(text=df.to_string(index=False), metadata={"sheet_name": sheet_name})
        )
        step.detail(f"{sheet_name:<12} {len(df):>3} 行 → 摘要 + 内容 各 1 份")

    step.note(f"{len(summary_docs)} 个 sheet")

# ---------------------------------------------------------------- 3. 摘要索引
with stage.step(
    "建摘要索引",
    api="VectorStoreIndex(summary_docs)",
    why="这一层只管「问题属于哪个年份」，所以文档都很短",
) as step:
    summary_index = VectorStoreIndex(summary_docs)
    step.note(f"{len(summary_docs)} 节点")

# ---------------------------------------------------------------- 4. 内容索引
with stage.step(
    "建内容索引",
    api="VectorStoreIndex(content_docs)",
    why="每个 sheet 的整张表作为一个文档，靠 metadata 里的 sheet_name 打标签",
) as step:
    content_index = VectorStoreIndex(content_docs)
    step.note(f"{len(content_docs)} 节点")

# ---------------------------------------------------------------- 5. 两步查询
with stage.step(
    "两步查询：路由 → 过滤检索",
    step1="VectorIndexRetriever(summary_index, top_k=1)",
    step2="VectorIndexRetriever(content_index, filters=ExactMatchFilter(sheet_name))",
    query=QUERY,
) as step:
    summary_retriever = VectorIndexRetriever(index=summary_index, similarity_top_k=1)
    retrieved_nodes = summary_retriever.retrieve(QUERY)

    if not retrieved_nodes:
        step.detail("未能找到相关的电影年份信息。")
        matched_sheet_name = None
    else:
        matched_sheet_name = retrieved_nodes[0].node.metadata["sheet_name"]
        step.detail(f"第一步·路由结果    → {matched_sheet_name}")

        content_retriever = VectorIndexRetriever(
            index=content_index,
            similarity_top_k=1,
            filters=MetadataFilters(
                filters=[ExactMatchFilter(key="sheet_name", value=matched_sheet_name)]
            ),
        )
        step.detail(f"第二步·过滤条件    sheet_name == '{matched_sheet_name}'")
        step.note(f"{matched_sheet_name}")

# ---------------------------------------------------------------- 6. 回答
with stage.step(
    "生成回答",
    why="只把过滤后的那张表交给 LLM，避免它被其它年份的数据干扰",
) as step:
    if matched_sheet_name is None:
        response = "抱歉，未能找到相关的电影年份信息。"
    else:
        query_engine = RetrieverQueryEngine.from_args(content_retriever)
        response = query_engine.query(QUERY)
    step.detail(f"最终回答：{response}")
    step.note("完成")

stage.summary()

print()
print("=" * 72)
print("  观察点：和 06 对比")
print("=" * 72)
print("1. 两层的「路由」逻辑一样：都在摘要层找到年份")
print("2. 第二层不同：06 让 Pandas 引擎算，07 只是把对应表格检索出来交给 LLM")
print("3. 07 更安全（不执行生成的代码），但算术问题更依赖模型发挥")
print("4. 关键点：MetadataFilters + ExactMatchFilter 是「结构化条件」和")
print("   「向量检索」结合的方式 —— 生产里很常用（先按租户/时间过滤，再算相似度）")
print()
print("提示：如果回答里的数字和表格对不上，说明 LLM 算错了，换成 06 的 Pandas 引擎更稳")
print("=" * 72)
