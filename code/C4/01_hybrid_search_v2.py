"""C4-01 · 混合检索 v2：SigLIP 出稠密、TF-IDF 出稀疏

对应文档：docs/chapter4/01_hybrid_search.md（与 01 对照阅读）

和 01 的差别在「谁产出两路向量」：

    01    BGE-M3：一个模型同时出 dense + sparse（稀疏来自模型学到的词权重）
    01v2  SigLIP + TF-IDF：SigLIP 出 dense，TF-IDF 出 sparse（稀疏来自统计量）

v2 的好处：TF-IDF 不需要训练、可解释性强、换语料也能直接用；
代价：稀疏那路由 sklearn 现算，和稠密那路不在同一个语义空间。

运行（要 Milvus，首次会下载 SigLIP 模型）：
    python3 code/C4/01_hybrid_search_v2.py
    RAG_LOG=quiet python3 code/C4/01_hybrid_search_v2.py

6 个阶段：
    [1/6] 连接 Milvus + 加载 SigLIP
    [2/6] 建 Collection + 两套索引
    [3/6] 生成双向量并入库
    [4/6] 查询编码
    [5/6] 单路检索对比
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
import torch
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
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModel, AutoProcessor

COLLECTION_NAME = "dragon_siglip_demo"
MILVUS_URI = "http://localhost:19530"
DATA_PATH = "../../data/C4/metadata/dragon.json"
SEARCH_QUERY = "悬崖上的巨龙"
SEARCH_FILTER = 'category in ["western_dragon", "chinese_dragon", "movie_character"]'
TOP_K = 5
OUTPUT_FIELDS = ["title", "path", "description", "category", "location", "environment"]

# 1. 初始化设置

# 2. 自定义SigLIP嵌入函数类
class SigLIPEmbeddingFunction:
    def __init__(self, model_name="google/siglip-base-patch16-256-multilingual", device="cpu"):
        """
        初始化SigLIP嵌入函数
        Args:
            model_name: SigLIP模型名称
            device: 设备类型 ("cpu" 或 "cuda")
        """
        self.model_name = model_name
        self.device = device
        
        print(f"--> 正在加载 SigLIP 模型: {model_name}")
        self.model = AutoModel.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        
        # 初始化TF-IDF作为稀疏向量生成器
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,  # 限制词汇表大小以节省空间
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.tfidf_fitted = False
        
        # 获取文本编码器的输出维度
        with torch.no_grad():
            dummy_text = ["test"]
            inputs = self.processor(text=dummy_text, padding="max_length", return_tensors="pt")
            outputs = self.model.text_model(**{k: v.to(device) for k, v in inputs.items() if k != 'pixel_values'})
            self.dense_dim = outputs.pooler_output.shape[-1]
        
        print(f"--> SigLIP 模型加载完成。密集向量维度: {self.dense_dim}")
    
    @property
    def dim(self):
        """返回维度信息，兼容原BGE-M3接口"""
        return {
            "dense": self.dense_dim,
            "sparse": self.tfidf_vectorizer.max_features if self.tfidf_fitted else 10000
        }
    
    def fit_sparse(self, docs):
        """拟合稀疏向量模型（TF-IDF）"""
        print("--> 正在拟合 TF-IDF 模型...")
        self.tfidf_vectorizer.fit(docs)
        self.tfidf_fitted = True
        print(f"--> TF-IDF 模型拟合完成。词汇表大小: {len(self.tfidf_vectorizer.vocabulary_)}")
    
    def encode_text_dense(self, texts):
        """使用SigLIP编码文本为密集向量"""
        if isinstance(texts, str):
            texts = [texts]
        
        dense_vectors = []
        batch_size = 8  # 减小批次大小以节省内存
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                inputs = self.processor(text=batch_texts, padding="max_length", truncation=True, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items() if k != 'pixel_values'}
                
                outputs = self.model.text_model(**inputs)
                embeddings = outputs.pooler_output
                
                # 归一化向量
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                dense_vectors.extend(embeddings.cpu().numpy())
        
        return np.array(dense_vectors)
    
    def encode_text_sparse(self, texts):
        """使用TF-IDF编码文本为稀疏向量"""
        if not self.tfidf_fitted:
            raise ValueError("请先调用 fit_sparse() 方法拟合TF-IDF模型")
        
        if isinstance(texts, str):
            texts = [texts]
        
        sparse_matrix = self.tfidf_vectorizer.transform(texts)
        return sparse_matrix
    
    def __call__(self, texts):
        """主调用方法，返回密集和稀疏向量"""
        if isinstance(texts, str):
            texts = [texts]
        
        # 如果还没有拟合稀疏模型，先拟合
        if not self.tfidf_fitted:
            self.fit_sparse(texts)
        
        dense_vectors = self.encode_text_dense(texts)
        sparse_vectors = self.encode_text_sparse(texts)
        
        return {
            "dense": dense_vectors,
            "sparse": sparse_vectors
        }

stage = Stage(total=6, title="C4 · 混合检索 v2（SigLIP + TF-IDF）")
stage.header("稠密来自 SigLIP，稀疏来自 TF-IDF 统计，再用 RRF 融合")

# 3. 连接 Milvus 并初始化嵌入模型
with stage.step(
    "连接 Milvus + 加载 SigLIP",
    uri=MILVUS_URI,
    model="google/siglip-base-patch16-256-multilingual",
    why="v2 的两路向量来源不同：SigLIP 是训练出来的，TF-IDF 是统计出来的",
) as step:
    connections.connect(uri=MILVUS_URI)
    ef = SigLIPEmbeddingFunction(device="cpu")  # 有 GPU 可改 "cuda"
    step.detail("SigLIP 负责稠密向量；类内部用 TfidfVectorizer 产出稀疏向量")
    step.note("已加载")

# 4. 创建 Collection
milvus_client = MilvusClient(uri=MILVUS_URI)
with stage.step(
    "建 Collection + 两套索引",
    name=COLLECTION_NAME,
    why="结构和 01 完全一样，只是向量来源换了：同一个 schema 可以换不同编码器",
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
    schema = CollectionSchema(fields, description="使用SigLIP的龙混合检索示例")
    collection = Collection(name=COLLECTION_NAME, schema=schema, consistency_level="Strong")

    collection.create_index("sparse_vector", {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"})
    step.detail("稀疏索引：SPARSE_INVERTED_INDEX / IP")
    collection.create_index("dense_vector", {"index_type": "AUTOINDEX", "metric_type": "IP"})
    step.detail("稠密索引：AUTOINDEX / IP")
    step.note("双索引就绪")

collection = Collection(COLLECTION_NAME)

# 6. 加载数据并插入
with stage.step(
    "生成双向量并入库",
    path=DATA_PATH,
    why="SigLIP 返回的稀疏矩阵要手动转成 Milvus 要的 dict 格式，这点和 BGE-M3 不同",
) as step:
    collection.load()
    step.detail("Collection 已加载到内存")

    if not collection.is_empty:
        step.detail(f"已有 {collection.num_entities} 条数据，跳过插入")
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
        step.detail("SigLIP + TF-IDF 编码完成")

        dense_vectors = embeddings["dense"].tolist()
        sparse_vectors = []
        sparse_matrix = embeddings["sparse"]
        for i in range(sparse_matrix.shape[0]):
            row = sparse_matrix.getrow(i)
            sparse_dict = {}
            for j in range(row.nnz):
                sparse_dict[row.indices[j]] = float(row.data[j])
            sparse_vectors.append(sparse_dict)

        collection.insert([
            [d["img_id"] for d in metadata],
            [d["path"] for d in metadata],
            [d["title"] for d in metadata],
            [d["description"] for d in metadata],
            [d["category"] for d in metadata],
            [d["location"] for d in metadata],
            [d["environment"] for d in metadata],
            sparse_vectors,
            dense_vectors,
        ])
        collection.flush()
        step.detail(f"flush 完成，总数 {collection.num_entities}")
        step.note(f"{collection.num_entities} 条")

# 7. 执行搜索
with stage.step(
    "查询编码",
    query=SEARCH_QUERY,
    filter=SEARCH_FILTER,
) as step:
    query_embeddings = ef([SEARCH_QUERY])
    dense_vec = query_embeddings["dense"][0].tolist()

    sparse_matrix = query_embeddings["sparse"]
    sparse_row = sparse_matrix.getrow(0)
    sparse_dict = {}
    for j in range(sparse_row.nnz):
        sparse_dict[sparse_row.indices[j]] = float(sparse_row.data[j])

    step.detail(f"稠密向量：dim={len(dense_vec)}，范数={np.linalg.norm(dense_vec):.4f}")
    step.detail(f"稀疏向量：dim={sparse_matrix.shape[1]}，非零 {sparse_row.nnz} 个")
    density = sparse_row.nnz / sparse_matrix.shape[1] * 100
    step.detail(f"稀疏密度约 {density:.6f}%")
    step.note(f"sparse nnz={sparse_row.nnz}")

    search_params = {"metric_type": "IP", "params": {}}

# 单路对比
with stage.step(
    "单路检索对比",
    dense="dense_vector (SigLIP)",
    sparse="sparse_vector (TF-IDF)",
    why="两路的分数量纲本来就不同，这也是为什么最后要用 RRF 而不是加权求和",
) as step:
    dense_results = collection.search(
        [dense_vec], anns_field="dense_vector", param=search_params,
        limit=TOP_K, expr=SEARCH_FILTER, output_fields=OUTPUT_FIELDS,
    )[0]
    step.detail("--- 稠密单独检索 ---")
    for i, hit in enumerate(dense_results, 1):
        step.detail(f"  {i}. {hit.entity.get('title')}  (Score: {hit.distance:.4f})")

    sparse_results = collection.search(
        [sparse_dict], anns_field="sparse_vector", param=search_params,
        limit=TOP_K, expr=SEARCH_FILTER, output_fields=OUTPUT_FIELDS,
    )[0]
    step.detail("")
    step.detail("--- 稀疏单独检索 ---")
    for i, hit in enumerate(sparse_results, 1):
        step.detail(f"  {i}. {hit.entity.get('title')}  (Score: {hit.distance:.4f})")
    step.note("2 路结果")

# 混合 + 清理
with stage.step(
    "RRF 混合检索 + 清理",
    reranker="RRFRanker(k=60)",
) as step:
    rerank = RRFRanker(k=60)
    dense_req = AnnSearchRequest([dense_vec], "dense_vector", search_params, limit=TOP_K)
    sparse_req = AnnSearchRequest([sparse_dict], "sparse_vector", search_params, limit=TOP_K)

    results = collection.hybrid_search(
        [sparse_req, dense_req], rerank=rerank,
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
print("  观察点：和 01（BGE-M3）对比")
print("=" * 72)
print("1. 01 的稀疏是模型学出来的词权重；01v2 的稀疏是 TF-IDF 统计出来的")
print("2. TF-IDF 依赖语料共现，语料小的时候区分度反而更稳、更可解释")
print("3. SigLIP 是图文对齐模型，用在纯文本检索上属于「顺带借用」")
print("4. 两种方案的 Collection schema 完全一样 —— 换编码器不影响存储结构")
print()
print("提示：SigLIP 首次运行要从 HF 下载模型，国内可设 HF_ENDPOINT=https://hf-mirror.com")
print("=" * 72)
