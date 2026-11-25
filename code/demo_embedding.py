"""把「文本/图片 → 向量」这一步单独拎出来看清楚。

C1~C4 里一共有 5 种编码方式，这个脚本逐个演示：输入什么、调哪个模型、输出多少维、长什么样。

    python3 code/demo_embedding.py            # 全部跑一遍
    python3 code/demo_embedding.py text       # 只跑某一种

对应关系（脚本 → 本章用到的编码方式）：
    C1/01, C1/02, C3/02, C3/03, C3/05~07, C4/02~06   → 1) bge 文本稠密向量
    C2/04                                            → 1) 同上（用于语义切分的断点判断）
    C3/01, C3/04, C3/work_*                          → 2) Visualized-BGE 图文联合向量
    C4/01, C4/text2sql                               → 3) BGE-M3 稠密 + 稀疏
    C4/01_v2                                         → 4) SigLIP 文本 + TF-IDF 稀疏
    C4/07                                            → 5) ColBERT 式 token 级向量（重排用）
"""
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

import bootstrap  # noqa: E402,F401  加载 code/.env、固定缓存

DATA = REPO_ROOT / "data"
MODELS = REPO_ROOT / "models"

TEXTS = ["悬崖上的巨龙", "一只可爱的小猫"]
IMG = DATA / "C3/dragon/dragon02.png"


def _show(name, dim, vec, note=""):
    head = ", ".join(f"{v:.4f}" for v in list(vec)[:5])
    print(f"    模型: {name}")
    print(f"    维度: {dim}")
    if note:
        print(f"    说明: {note}")
    print(f"    前5维: [{head}, …]")


def demo_bge_text():
    print("\n[1] bge 文本稠密向量  （C1/C2/C3/C4 里用得最多的一路）")
    from langchain_huggingface import HuggingFaceEmbeddings

    emb = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    docs_vec = emb.embed_documents(TEXTS)   # 建库用：一批文本
    query_vec = emb.embed_query(TEXTS[0])   # 查询用：单条

    _show("BAAI/bge-small-zh-v1.5", len(query_vec), query_vec,
          "encode_kwargs={'normalize_embeddings': True} → L2 归一化，配合 COSINE/内积")

    # 相似度：归一化后内积 = 余弦相似度
    import numpy as np

    a, b = np.array(docs_vec[0]), np.array(docs_vec[1])
    print(f"    '悬崖上的巨龙' vs '一只可爱的小猫' 余弦相似度 = {float(a @ b):.4f}")


def demo_visualized_bge():
    print("\n[2] Visualized-BGE 图文联合向量  （C3/01、C3/04、C3/work_*）")
    import torch

    if str(CODE_DIR / "C3") not in sys.path:
        sys.path.insert(0, str(CODE_DIR / "C3"))
    from visual_bge.visual_bge.modeling import Visualized_BGE

    model = Visualized_BGE(
        model_name_bge="BAAI/bge-base-en-v1.5",
        model_weight=str(MODELS / "bge/Visualized_base_en_v1.5.pth"),
    )
    model.eval()
    with torch.no_grad():
        t = model.encode(text="datawhale开源组织的logo")     # 只文本
        i = model.encode(image=str(IMG))                      # 只图片
        ti = model.encode(image=str(IMG), text="悬崖上的白龙")  # 图文拼一起

    print("    三种入参都落在同一个 768 维空间里，可直接互相算相似度：")
    print(f"      纯文本      shape={tuple(t.shape)}")
    print(f"      纯图片      shape={tuple(i.shape)}")
    print(f"      图片+文本   shape={tuple(ti.shape)}")
    print(f"    文本·图片 相似度 = {float(t @ i.T):.4f}")
    print(f"    图片·(图+文) 相似度 = {float(i @ ti.T):.4f}")


def demo_bge_m3():
    print("\n[3] BGE-M3 稠密 + 稀疏  （C4/01、C4/text2sql/knowledge_base）")
    import numpy as np
    from pymilvus.model.hybrid import BGEM3EmbeddingFunction

    ef = BGEM3EmbeddingFunction(use_fp16=False, device="cpu")
    out = ef(TEXTS)
    dense, sparse = out["dense"], out["sparse"]
    row = sparse._getrow(0)
    _show("BAAI/bge-m3（dense）", len(dense[0]), dense[0])
    print(f"    sparse: 词表维度={sparse.shape[1]}，本条非零项={row.nnz}"
          f"（密度 {row.nnz / sparse.shape[1] * 100:.6f}%）")
    print(f"    sparse 前3个非零项: "
          f"{[(int(k), round(float(v), 4)) for k, v in list(zip(row.indices[:3], row.data[:3]))]}")
    print("    稀疏向量的下标是 BGE-M3 词表里的 token id，值是该 token 的权重")


def demo_siglip_tfidf():
    print("\n[4] SigLIP 文本向量 + TF-IDF 稀疏  （C4/01_v2）")
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from transformers import AutoModel, AutoProcessor
    import torch

    name = "google/siglip-base-patch16-256-multilingual"
    model = AutoModel.from_pretrained(name)
    proc = AutoProcessor.from_pretrained(name)
    model.eval()
    with torch.no_grad():
        inputs = proc(text=TEXTS, padding="max_length", truncation=True, return_tensors="pt")
        inputs = {k: v for k, v in inputs.items() if k != "pixel_values"}
        vec = model.text_model(**inputs).pooler_output
        vec = torch.nn.functional.normalize(vec, p=2, dim=1)
    _show(name + "（text_model.pooler_output）", vec.shape[1], vec[0].tolist())

    tf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    mat = tf.fit_transform(TEXTS)
    row = mat.getrow(0)
    print(f"    TF-IDF 稀疏: 词表={mat.shape[1]}，本条非零项={row.nnz}")
    print(f"    非零项下标前3个: {row.indices[:3].tolist()}"
          f"（对应词: {[tf.get_feature_names_out()[i] for i in row.indices[:3]]}）")


def demo_colbert():
    print("\n[5] ColBERT 式 token 级向量  （C4/07 重排器实际用的做法）")
    import torch
    from transformers import AutoModel, AutoTokenizer
    import torch.nn.functional as F

    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")
    model.eval()
    inputs = tok(["a dragon on the cliff"], return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        hidden = model(**inputs).last_hidden_state
        hidden = F.normalize(hidden, p=2, dim=-1)
    print(f"    模型: bert-base-uncased")
    print(f"    输出: [batch, token数, 768] = {tuple(hidden.shape)}")
    print(f"    说明: 一句话不是一个向量，而是每个 token 一个 768 维向量；")
    print(f"          重排时对每个 query token 取 max 相似度再求和（MaxSim）")


DEMOS = {
    "text": demo_bge_text,
    "visual": demo_visualized_bge,
    "bgem3": demo_bge_m3,
    "siglip": demo_siglip_tfidf,
    "colbert": demo_colbert,
}


def main():
    which = sys.argv[1:] or list(DEMOS)
    for key in which:
        if key not in DEMOS:
            print(f"未知选项 {key}，可选: {list(DEMOS)}")
            continue
        DEMOS[key]()


if __name__ == "__main__":
    main()
