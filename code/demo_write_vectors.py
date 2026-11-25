"""往 Milvus 里写一批「可留存的」向量，配合 code/inspect_vectors.py 观察存取的数据。

教程里的 C3/04、C4/01 等脚本跑完会 drop_collection，所以平时去看 Milvus 是空的。
这个脚本用三种典型编码方式各建一个 collection，跑完不删，方便你去看：

    demo_text_dense     BAAI/bge-small-zh-v1.5        纯文本 → 512 维稠密向量
    demo_hybrid_bgem3   BAAI/bge-m3                   文本 → 1024 维稠密 + 稀疏向量
    demo_multimodal     Visualized-BGE（bge-base-en）  图片/图文 → 768 维
    demo_tfidf_sparse   sklearn TfidfVectorizer       文本 → 稀疏向量（C4/01_v2 的做法）

用法：

    python3 code/demo_write_vectors.py            # 写入/覆盖
    python3 code/inspect_vectors.py --milvus      # 查看
    python3 code/demo_write_vectors.py --clean    # 全部删掉
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

import bootstrap  # noqa: E402,F401  加载 code/.env、固定缓存、修代理与证书

MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
DATA_DIR = REPO_ROOT / "data"
MODEL_DIR = REPO_ROOT / "models"

TEXTS = [
    "张三是法外狂徒",
    "FAISS 是一个用于高效相似性搜索和密集向量聚类的库。",
    "LangChain 是一个用于开发由语言模型驱动的应用程序的框架。",
]

ALL_COLLECTIONS = ["demo_text_dense", "demo_hybrid_bgem3", "demo_multimodal", "demo_tfidf_sparse"]


def drop_all(client):
    for name in ALL_COLLECTIONS:
        if client.has_collection(name):
            client.drop_collection(name)
            print(f"已删除 collection: {name}")


def write_text_dense(client):
    """最基础的一路：纯文本 → 稠密向量（C1、C3 里最常用的做法）。"""
    from langchain_huggingface import HuggingFaceEmbeddings

    name = "demo_text_dense"
    emb = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectors = emb.embed_documents(TEXTS)
    dim = len(vectors[0])

    if client.has_collection(name):
        client.drop_collection(name)
    client.create_collection(
        collection_name=name,
        dimension=dim,
        metric_type="COSINE",
        auto_id=False,
        id_type="int",
    )
    client.insert(
        collection_name=name,
        data=[{"id": i, "vector": v, "text": t} for i, (v, t) in enumerate(zip(vectors, TEXTS))],
    )
    # 必须 flush，否则外部工具（Attu / get_collection_stats）看到 row_count 还是 0
    client.flush(collection_name=name)
    print(f"[{name}] bge-small-zh-v1.5  dim={dim}  rows={len(vectors)}")


def write_hybrid_bgem3(client):
    """稠密 + 稀疏双向量（C4/01 的混合检索就是这么存的）。"""
    from pymilvus import DataType
    from pymilvus.model.hybrid import BGEM3EmbeddingFunction

    name = "demo_hybrid_bgem3"
    docs = [d["title"] + " " + d["description"] for d in json.loads((DATA_DIR / "C4/metadata/dragon.json").read_text())]

    ef = BGEM3EmbeddingFunction(use_fp16=False, device="cpu")
    out = ef(docs)
    dense = out["dense"]
    sparse = out["sparse"]
    dim = len(dense[0])

    if client.has_collection(name):
        client.drop_collection(name)

    from pymilvus import Collection, CollectionSchema, FieldSchema

    schema = CollectionSchema(
        [
            FieldSchema("id", DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema("text", DataType.VARCHAR, max_length=4096),
            FieldSchema("dense", DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema("sparse", DataType.SPARSE_FLOAT_VECTOR),
        ]
    )
    col = Collection(name, schema=schema, using="default", consistency_level="Strong")
    col.create_index("dense", {"index_type": "AUTOINDEX", "metric_type": "IP"})
    col.create_index("sparse", {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"})
    col.insert(
        [
            list(range(len(docs))),
            docs,
            [d.tolist() for d in dense],
            [sparse._getrow(i) for i in range(len(docs))],
        ]
    )
    col.flush()
    print(f"[{name}] bge-m3  dense={dim}  sparse_vocab={sparse.shape[1]}  rows={len(docs)}")


def write_multimodal(client):
    """图片/图文 → 同一个 768 维空间（C3/01、C3/04 的做法）。"""
    import torch

    if str(CODE_DIR / "C3") not in sys.path:
        sys.path.insert(0, str(CODE_DIR / "C3"))
    from visual_bge.visual_bge.modeling import Visualized_BGE

    name = "demo_multimodal"
    meta = {m["path"].split("/")[-1]: m for m in json.loads((DATA_DIR / "C4/metadata/dragon.json").read_text())}
    images = sorted(glob.glob(str(DATA_DIR / "C3/dragon/*.png")))

    model = Visualized_BGE(
        model_name_bge="BAAI/bge-base-en-v1.5",
        model_weight=str(MODEL_DIR / "bge/Visualized_base_en_v1.5.pth"),
    )
    model.eval()

    rows = []
    with torch.no_grad():
        for i, path in enumerate(images):
            base = os.path.basename(path)
            m = meta.get(base, {})
            caption = m.get("title", "")
            # 有描述时做「图文联合编码」，没有就只编码图片
            if caption:
                vec = model.encode(image=path, text=caption).tolist()[0]
            else:
                vec = model.encode(image=path).tolist()[0]
            rows.append(
                {"id": i, "image_path": f"data/C3/dragon/{base}", "caption": caption, "vector": vec}
            )

    dim = len(rows[0]["vector"])
    if client.has_collection(name):
        client.drop_collection(name)
    client.create_collection(
        collection_name=name, dimension=dim, metric_type="COSINE", auto_id=False, id_type="int"
    )
    client.insert(collection_name=name, data=rows)
    client.flush(collection_name=name)
    print(f"[{name}] Visualized-BGE  dim={dim}  rows={len(rows)}")


def write_tfidf_sparse(client):
    """C4/01_v2 用 TF-IDF 手工造稀疏向量的做法（维度=词表大小）。"""
    from pymilvus import DataType
    from sklearn.feature_extraction.text import TfidfVectorizer

    name = "demo_tfidf_sparse"
    docs = [d["title"] + " " + d["description"] for d in json.loads((DATA_DIR / "C4/metadata/dragon.json").read_text())]

    vec = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    mat = vec.fit_transform(docs)

    if client.has_collection(name):
        client.drop_collection(name)

    from pymilvus import Collection, CollectionSchema, FieldSchema

    schema = CollectionSchema(
        [
            FieldSchema("id", DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema("text", DataType.VARCHAR, max_length=4096),
            FieldSchema("sparse", DataType.SPARSE_FLOAT_VECTOR),
        ]
    )
    col = Collection(name, schema=schema, using="default", consistency_level="Strong")
    col.create_index("sparse", {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"})
    col.insert(
        [
            list(range(len(docs))),
            docs,
            [
                {int(j): float(mat[i, j]) for j in mat[i].indices}
                for i in range(mat.shape[0])
            ],
        ]
    )
    col.flush()
    print(f"[{name}] TF-IDF  vocab={mat.shape[1]}  rows={mat.shape[0]}")


def main():
    ap = argparse.ArgumentParser(description="往 Milvus 写入可留存的演示向量")
    ap.add_argument("--clean", action="store_true", help="删除全部演示 collection 后退出")
    ap.add_argument("--only", choices=["text", "hybrid", "multimodal", "tfidf"], help="只写某一种")
    args = ap.parse_args()

    from pymilvus import MilvusClient, connections

    # ORM Collection 走的是 connections 里的默认连接，MilvusClient 是另一套；
    # 两种接口这里都要用，所以都建一下
    connections.connect(uri=MILVUS_URI)
    client = MilvusClient(uri=MILVUS_URI)
    print(f"Milvus: {MILVUS_URI}  现有 collection: {client.list_collections()}")

    if args.clean:
        drop_all(client)
        return

    if args.only in (None, "text"):
        write_text_dense(client)
    if args.only in (None, "hybrid"):
        write_hybrid_bgem3(client)
    if args.only in (None, "multimodal"):
        write_multimodal(client)
    if args.only in (None, "tfidf"):
        write_tfidf_sparse(client)

    print(f"\n完成。现有 collection: {client.list_collections()}")
    print("查看数据：python3 code/inspect_vectors.py --milvus")


if __name__ == "__main__":
    main()
