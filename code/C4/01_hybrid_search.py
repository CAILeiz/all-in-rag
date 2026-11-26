"""C4-01 · 混合检索：稠密向量 + 稀疏向量，用 RRF 融合

对应文档：docs/chapter4/01_hybrid_search.md

这一节的要点：**一路检索永远有盲区，两路互补才稳。**

    DENSE（稠密）  语义相近就能召回，但「中华金龙」这种专有名词容易漏
    SPARSE（稀疏） 按词的权重匹配，专有名词命中准，但不懂同义词
    HYBRID       两路各出 Top-K，用 RRF 按「名次」融合

BGE-M3 的好处是一次前向同时产出两种向量，不用维护两套模型。

前置条件：Milvus 在跑（docker compose -f code/docker-compose.local.yml up -d）

运行：
    python3 code/C4/01_hybrid_search.py
    RAG_LOG=quiet python3 code/C4/01_hybrid_search.py

6 个阶段：
    [1/6] 连接 Milvus + 加载 BGE-M3
    [2/6] 建 Collection + 两套索引
    [3/6] 读数据、生成双向量、入库
    [4/6] 查询编码（同时看稠密/稀疏向量长什么样）
    [5/6] 单路检索对比（稠密 / 稀疏）
    [6/6] RRF 混合检索 + 清理
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

import json
import os

import numpy as np
from pymilvus import (
    AnnSearchRequest,
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusClient,
    RRFRanker,
    connections,
)
from pymilvus.model.hybrid import BGEM3EmbeddingFunction

COLLECTION_NAME = "dragon_hybrid_demo"
MILVUS_URI = "http://localhost:19530"
DATA_PATH = "../../data/C4/metadata/dragon.json"
SEARCH_QUERY = "悬崖上的巨龙"
SEARCH_FILTER = 'category in ["western_dragon", "chinese_dragon", "movie_character"]'
TOP_K = 5
OUTPUT_FIELDS = ["title", "path", "description", "category", "location", "environment"]

stage = Stage(total=6, title="C4 · 混合检索（BGE-M3 稠密+稀疏）")
stage.header("一路向量检索有盲区，两路互补 + RRF 融合")

# ---------------------------------------------------------------- 1. 连接 + 模型
with stage.step(
    "连接 Milvus + 加载 BGE-M3",
    uri=MILVUS_URI,
    why="BGE-M3 一次前向同时产出 dense 和 sparse 两种向量，不必维护两套模型",
) as step:
    connections.connect(uri=MILVUS_URI)
    ef = BGEM3EmbeddingFunction(use_fp16=False, device="cpu")
    step.detail(f"稠密向量维度：{ef.dim['dense']}")
    step.detail("稀疏向量维度：词表大小（按词给权重，长度不固定）")
    step.note(f"dense={ef.dim['dense']}")

# ---------------------------------------------------------------- 2. Collection
milvus_client = MilvusClient(uri=MILVUS_URI)
with stage.step(
    "建 Collection + 两套索引",
    name=COLLECTION_NAME,
    why="稠密用 AUTOINDEX（面向稠密向量），稀疏用 SPARSE_INVERTED_INDEX（倒排表）",
) as step:
    if milvus_client.has_collection(COLLECTION_NAME):
        milvus_client.drop_collection(COLLECTION_NAME)
        step.detail("已删除同名旧 Collection")

    fields = [
        FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100),
        FieldSchema(name="img_id", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="path", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=4096),
        FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="location", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="environment", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
        FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=ef.dim["dense"]),
    ]

    schema = CollectionSchema(fields, description="关于龙的混合检索示例")
    collection = Collection(name=COLLECTION_NAME, schema=schema, consistency_level="Strong")
    step.detail("10 个字段，其中 2 个是向量字段")

    collection.create_index("sparse_vector", {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"})
    step.detail("稀疏索引：SPARSE_INVERTED_INDEX / IP")
    collection.create_index("dense_vector", {"index_type": "AUTOINDEX", "metric_type": "IP"})
    step.detail("稠密索引：AUTOINDEX / IP")
    step.note("双索引就绪")

collection = Collection(COLLECTION_NAME)

# ---------------------------------------------------------------- 3. 入库
with stage.step(
    "生成双向量并入库",
    path=DATA_PATH,
    why="标题+描述+地点+环境拼成一段文本，一次编码出 dense 和 sparse",
) as step:
    collection.load()
    step.detail("Collection 已加载到内存")

    if not collection.is_empty:
        step.detail(f"Collection 已有 {collection.num_entities} 条数据，跳过插入")
        step.note(f"{collection.num_entities} 条（已有）")
    else:
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(f"数据文件未找到: {DATA_PATH}")
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        docs, metadata = [], []
        for item in dataset:
            parts = [
                item.get("title", ""),
                item.get("description", ""),
                item.get("location", ""),
                item.get("environment", ""),
            ]
            docs.append(" ".join(filter(None, parts)))
            metadata.append(item)
        step.detail(f"读入 {len(docs)} 条记录")

        embeddings = ef(docs)
        step.detail("双向量生成完成")

        collection.insert([
            [d["img_id"] for d in metadata],
            [d["path"] for d in metadata],
            [d["title"] for d in metadata],
            [d["description"] for d in metadata],
            [d["category"] for d in metadata],
            [d["location"] for d in metadata],
            [d["environment"] for d in metadata],
            embeddings["sparse"],
            embeddings["dense"],
        ])
        collection.flush()
        step.detail(f"flush 完成，总数 {collection.num_entities}")
        step.note(f"{collection.num_entities} 条")

# ---------------------------------------------------------------- 4. 查询编码
with stage.step(
    "查询编码",
    query=SEARCH_QUERY,
    filter=SEARCH_FILTER,
    why="同一条 query 也出两路向量；expr 先按 category 做结构化过滤",
) as step:
    query_embeddings = ef([SEARCH_QUERY])
    dense_vec = query_embeddings["dense"][0]
    sparse_vec = query_embeddings["sparse"]._getrow(0)

    step.detail(f"稠密向量：dim={len(dense_vec)}，范数={np.linalg.norm(dense_vec):.4f}")
    step.detail(f"稀疏向量：dim={sparse_vec.shape[1]}，非零 {sparse_vec.nnz} 个")
    density = sparse_vec.nnz / sparse_vec.shape[1] * 100
    step.detail(f"稀疏密度约 {density:.6f}% —— 绝大多数位置是 0，所以用倒排存")
    step.note(f"sparse nnz={sparse_vec.nnz}")

    search_params = {"metric_type": "IP", "params": {}}

# ---------------------------------------------------------------- 5. 单路对比
with stage.step(
    "单路检索对比",
    dense="dense_vector (AUTOINDEX/IP)",
    sparse="sparse_vector (SPARSE_INVERTED_INDEX/IP)",
    why="先分别看两路各自的排序，才能理解 RRF 融合后为什么变了",
) as step:
    dense_results = collection.search(
        [dense_vec], anns_field="dense_vector", param=search_params,
        limit=TOP_K, expr=SEARCH_FILTER, output_fields=OUTPUT_FIELDS,
    )[0]
    step.detail("--- 稠密单独检索 ---")
    for i, hit in enumerate(dense_results, 1):
        step.detail(f"  {i}. {hit.entity.get('title')}  (Score: {hit.distance:.4f})")

    sparse_results = collection.search(
        [sparse_vec], anns_field="sparse_vector", param=search_params,
        limit=TOP_K, expr=SEARCH_FILTER, output_fields=OUTPUT_FIELDS,
    )[0]
    step.detail("")
    step.detail("--- 稀疏单独检索 ---")
    for i, hit in enumerate(sparse_results, 1):
        step.detail(f"  {i}. {hit.entity.get('title')}  (Score: {hit.distance:.4f})")
    step.note("2 路结果")

# ---------------------------------------------------------------- 6. 混合 + 清理
with stage.step(
    "RRF 混合检索 + 清理",
    reranker="RRFRanker(k=60)",
    why="RRF 只看名次不看分数，所以两路分数量纲不同也能直接融合",
) as step:
    dense_req = AnnSearchRequest([dense_vec], "dense_vector", search_params, limit=TOP_K)
    sparse_req = AnnSearchRequest([sparse_vec], "sparse_vector", search_params, limit=TOP_K)

    results = collection.hybrid_search(
        [sparse_req, dense_req], rerank=RRFRanker(k=60),
        limit=TOP_K, output_fields=OUTPUT_FIELDS,
    )[0]

    step.detail("--- 混合检索结果 ---")
    for i, hit in enumerate(results, 1):
        step.detail(f"  {i}. {hit.entity.get('title')}  (Score: {hit.distance:.4f})")
        step.detail(f"     描述：{hit.entity.get('description')[:60]}...")

    milvus_client.release_collection(collection_name=COLLECTION_NAME)
    milvus_client.drop_collection(COLLECTION_NAME)
    step.detail(f"已释放并删除 Collection: {COLLECTION_NAME}")
    step.note(f"Top{len(results)} + 已清理")

stage.summary()

print()
print("=" * 72)
print("  观察点")
print("=" * 72)
print("1. 稀疏向量密度极低（万分之一级），所以必须用倒排索引，不能用普通向量索引")
print("2. 稠密看语义、稀疏看词，两路的 Top5 通常不完全一样")
print("3. RRF 的分数不是相似度，是「1/(k+名次)」的累加，所以别拿它和 IP 分数比大小")
print("4. expr 过滤器在检索前生效，能先砍掉一半候选，对性能帮助很大")
print()
print("提示：BGE-M3 首次运行要装 tokenizers 等依赖，约 1-2 分钟")
print("=" * 72)
