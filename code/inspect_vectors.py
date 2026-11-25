"""向量数据库「存取的数据」巡检工具。

支持四类存储，C1~C4 里用到的都在里面：

    Milvus      C3/04、C3/work_*、C4/01、C4/01_v2、C4/text2sql   （服务端，localhost:19530）
    FAISS       C3/02、C4/07、C4/work_rerank                     （本地 index.faiss + index.pkl）
    LlamaIndex  C3/03、C3/05、C3/06、C3/07                       （本地 json）
    Chroma      C4/02、C4/04                                     （本地 sqlite，脚本里没落盘）

用法：

    # 看 Milvus 里所有 collection（自动列出 schema / 索引 / 行数 / 样本向量）
    python3 code/inspect_vectors.py --milvus

    # 只看某个 collection 的前 5 条，向量只打印前 8 维
    python3 code/inspect_vectors.py --milvus dragon_hybrid_demo -n 5 --head 8

    # 把某条向量的全部数值导成 json
    python3 code/inspect_vectors.py --milvus dragon_hybrid_demo -n 2 --dump out.json

    # 本地 FAISS
    python3 code/inspect_vectors.py --faiss code/C3/faiss_index_store

    # 本地 LlamaIndex 持久化目录
    python3 code/inspect_vectors.py --llamaindex code/C3/llamaindex_index_store

    # 什么都不带：把上面三类本地存储 + Milvus 都扫一遍
    python3 code/inspect_vectors.py
"""
import argparse
import json
import os
import pickle
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

import bootstrap  # noqa: E402,F401  加载 code/.env、修代理与证书

MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")

# ---------------------------------------------------------------- 打印工具

def _fmt_vec(vec, head=6, as_list=False):
    """向量只打印前 head 项，避免刷屏。

    - 稠密向量（list / ndarray）：按维打印
    - 稀疏向量（Milvus 返回 {index: value}）：按「非零项」打印，dim 报告的是非零个数
    """
    if as_list:
        return vec
    if isinstance(vec, dict):
        items = sorted(vec.items())
        preview = ", ".join(f"{k}:{v:.4f}" for k, v in items[:head])
        more = f", … (+{len(items) - head} 项)" if len(items) > head else ""
        return f"{{{preview}{more}}}  nnz={len(items)}"
    try:
        vals = list(vec)
    except TypeError:
        return vec
    preview = ", ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in vals[:head])
    more = f", … (+{len(vals) - head} 维)" if len(vals) > head else ""
    return f"[{preview}{more}]  dim={len(vals)}"


def _fmt_row(row: dict, dim_fields: dict, head=6):
    """dim_fields: {字段名: 维度}，只对向量字段做截断。"""
    out = {}
    for k, v in row.items():
        if k in dim_fields:
            out[k] = _fmt_vec(v, head)
        elif isinstance(v, str) and len(v) > 120:
            out[k] = v[:120] + "…"
        else:
            out[k] = v
    return out


def _section(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


# ---------------------------------------------------------------- Milvus

def _sparse_dim(v):
    """稀疏向量在 pymilvus 里是 {index: value} 字典。"""
    return len(v) if isinstance(v, dict) else None


def inspect_milvus(target: str | None, limit: int, head: int, dump: str | None):
    from pymilvus import DataType, MilvusClient

    client = MilvusClient(uri=MILVUS_URI)
    names = client.list_collections()

    if not names:
        _section(f"Milvus {MILVUS_URI}")
        print("（当前没有任何 collection）")
        print("提示：C3/04、C4/01 等脚本跑完会 drop_collection；")
        print("     想留下数据看，先跑  python3 code/demo_write_vectors.py")
        return

    if target:
        if target not in names:
            print(f"没有名为 {target!r} 的 collection；现有：{names}")
            return
        names = [target]

    for name in names:
        _section(f"Milvus collection: {name}")
        desc = client.describe_collection(collection_name=name)

        # get_collection_stats 在未 flush 时可能返回 0，这里用 count(*) 兜底
        try:
            row_count = client.query(
                collection_name=name, filter="", output_fields=["count(*)"]
            )[0]["count(*)"]
        except Exception:  # noqa: BLE001
            row_count = client.get_collection_stats(collection_name=name).get("row_count")

        print(f"行数: {row_count}")
        print(f"描述: {desc.get('description', '')}")
        print(f"consistency: {desc.get('consistency_level')}")
        print("\n字段:")
        vec_fields = {}
        for f in desc["fields"]:
            dtype = f["type"]
            dtype_name = dtype.name if hasattr(dtype, "name") else str(dtype)
            extra = ""
            if dtype in (DataType.FLOAT_VECTOR, DataType.BINARY_VECTOR):
                vec_fields[f["name"]] = f["params"]["dim"]
                extra = f" dim={f['params']['dim']}"
            elif dtype == DataType.SPARSE_FLOAT_VECTOR:
                vec_fields[f["name"]] = None  # 维度不固定
                extra = "（稀疏，维度=词表大小）"
            elif dtype == DataType.VARCHAR:
                extra = f" max_length={f['params'].get('max_length')}"
            pk = "  [主键]" if f.get("is_primary") else ""
            auto = "  auto_id" if f.get("auto_id") else ""
            print(f"  - {f['name']:<22} {dtype_name:<20}{extra}{pk}{auto}")

        indexes = client.list_indexes(collection_name=name)
        if indexes:
            print("\n索引:")
            for idx in indexes:
                try:
                    info = client.describe_index(collection_name=name, index_name=idx)
                    print(f"  - {idx}: {json.dumps({k: v for k, v in info.items() if k != 'index_name'})}")
                except Exception as e:  # noqa: BLE001
                    print(f"  - {idx}: (读取失败 {e})")

        try:
            client.load_collection(collection_name=name)
        except Exception:  # noqa: BLE001
            pass

        rows = client.query(
            collection_name=name,
            filter="",
            output_fields=["*"],
            limit=limit,
        )
        print(f"\n前 {len(rows)} 条数据（向量只显示前 {head} 维）:")
        for i, row in enumerate(rows, 1):
            # 稀疏向量维度用一个占位（词表大小），只报告非零个数
            dims = {k: (v if v is not None else f"nnz={_sparse_dim(row.get(k))}") for k, v in vec_fields.items()}
            print(f"\n  [{i}]")
            for k, v in _fmt_row(row, dims, head).items():
                print(f"      {k}: {v}")

        if dump:
            out = Path(dump)
            out.write_text(json.dumps(rows, ensure_ascii=False, indent=2, default=str))
            print(f"\n已导出 {len(rows)} 条原始数据到 {out}")


# ---------------------------------------------------------------- FAISS

def inspect_faiss(path: str, limit: int, head: int):
    import faiss
    import numpy as np

    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    idx_file = p / "index.faiss" if p.is_dir() else p
    if not idx_file.exists():
        print(f"跳过 FAISS：找不到 {idx_file}")
        return

    _section(f"FAISS 索引: {idx_file}")
    index = faiss.read_index(str(idx_file))
    print(f"向量条数 ntotal: {index.ntotal}")
    print(f"向量维度 d:      {index.d}")
    print(f"索引类型:        {type(index).__name__}")

    vecs = index.reconstruct_n(0, min(index.ntotal, limit))
    print(f"\n前 {len(vecs)} 条向量（前 {head} 维）:")
    for i, v in enumerate(vecs, 1):
        if isinstance(v, np.ndarray) and v.dtype != np.uint8:
            print(f"  [{i}] {_fmt_vec(v.tolist(), head)}  L2范数={float(np.linalg.norm(v)):.4f}")
        else:
            print(f"  [{i}] {v}")

    pkl = idx_file.parent / "index.pkl"
    if pkl.exists():
        with open(pkl, "rb") as f:
            docstore, index_to_id = pickle.load(f)
        print(f"\n对应的原始文本/元数据（{pkl.name}）:")
        for i, doc_id in enumerate(list(index_to_id.values())[:limit], 1):
            doc = docstore.search(doc_id)
            text = getattr(doc if not isinstance(doc, list) else doc[0], "page_content", None)
            meta = getattr(doc if not isinstance(doc, list) else doc[0], "metadata", {})
            print(f"  [{i}] id={doc_id}  meta={meta}")
            print(f"      text={text!r}")


# ---------------------------------------------------------------- LlamaIndex

def inspect_llamaindex(path: str, limit: int, head: int):
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    if not p.exists():
        print(f"跳过 LlamaIndex：找不到 {p}")
        return

    _section(f"LlamaIndex 持久化目录: {p}")
    files = sorted(x.name for x in p.glob("*.json"))
    print(f"文件清单: {files}")

    vs_file = p / "default__vector_store.json"
    if not vs_file.exists():
        print("没有 default__vector_store.json，跳过")
        return

    data = json.loads(vs_file.read_text())
    emb = data.get("embedding_dict", {})
    print(f"向量条数: {len(emb)}")
    if emb:
        first = next(iter(emb.values()))
        print(f"向量维度: {len(first)}")

    # 把 node_id -> 原文抓出来对照
    docstore = {}
    ds_file = p / "docstore.json"
    if ds_file.exists():
        ds = json.loads(ds_file.read_text())
        for node_id, node in (ds.get("docstore/data") or {}).items():
            try:
                docstore[node_id] = node["__data__"]["text"]
            except Exception:  # noqa: BLE001
                pass

    print(f"\n前 {min(len(emb), limit)} 条（向量前 {head} 维）:")
    for i, (node_id, vec) in enumerate(list(emb.items())[:limit], 1):
        text = docstore.get(node_id, "(原文在 docstore 里找不到)")
        print(f"\n  [{i}] node_id={node_id}")
        print(f"      {_fmt_vec(vec, head)}")
        print(f"      text={text[:160]!r}")


# ---------------------------------------------------------------- Chroma

def inspect_chroma(path: str, limit: int, head: int):
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    if not p.exists() or not any(p.iterdir()):
        print(f"跳过 Chroma：{p} 不存在或是空的")
        print("（C4/02、C4/04 用的是 Chroma.from_documents(...) 内存模式，没传 persist_directory，关掉进程就没了）")
        return

    _section(f"Chroma 持久化目录: {p}")
    try:
        import chromadb
    except ImportError:
        print("未安装 chromadb，跳过")
        return
    client = chromadb.PersistentClient(path=str(p))
    for col in client.list_collections():
        name = col.name if hasattr(col, "name") else str(col)
        c = client.get_collection(name)
        got = c.get(limit=limit, include=["embeddings", "documents", "metadatas"])
        print(f"\ncollection: {name}  条数={c.count()}")
        for i, doc_id in enumerate(got["ids"], 1):
            vec = got["embeddings"][i - 1]
            print(f"  [{i}] id={doc_id}  {_fmt_vec(vec, head)}")
            print(f"      text={str(got['documents'][i - 1])[:120]!r}")
            print(f"      meta={got['metadatas'][i - 1]}")


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(
        description="巡检向量数据库里实际存了什么（Milvus / FAISS / LlamaIndex / Chroma）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--milvus", nargs="?", const="", metavar="COLLECTION",
                    help="看 Milvus；可跟 collection 名，不跟则列出全部")
    ap.add_argument("--faiss", metavar="PATH", help="FAISS 目录（含 index.faiss/index.pkl）")
    ap.add_argument("--llamaindex", metavar="PATH", help="LlamaIndex persist_dir")
    ap.add_argument("--chroma", metavar="PATH", help="Chroma persist_directory")
    ap.add_argument("-n", "--limit", type=int, default=3, help="每个库取几条样本，默认 3")
    ap.add_argument("--head", type=int, default=6, help="向量打印前几维，默认 6")
    ap.add_argument("--dump", metavar="FILE", help="把 Milvus 查询结果原样导出为 json")
    args = ap.parse_args()

    touched = False
    if args.milvus is not None:
        inspect_milvus(args.milvus or None, args.limit, args.head, args.dump)
        touched = True
    if args.faiss:
        inspect_faiss(args.faiss, args.limit, args.head)
        touched = True
    if args.llamaindex:
        inspect_llamaindex(args.llamaindex, args.limit, args.head)
        touched = True
    if args.chroma:
        inspect_chroma(args.chroma, args.limit, args.head)
        touched = True

    if not touched:  # 全扫
        inspect_milvus(None, args.limit, args.head, args.dump)
        inspect_faiss(str(CODE_DIR / "C3" / "faiss_index_store"), args.limit, args.head)
        inspect_llamaindex(str(CODE_DIR / "C3" / "llamaindex_index_store"), args.limit, args.head)
        inspect_chroma(str(REPO_ROOT / ".cache" / "chroma"), args.limit, args.head)


if __name__ == "__main__":
    main()
