# 三个实用工具与说明

对应问题：①curl 调 deepseek-chat ②怎么看向量库里存的数据 ③测试里到底用了哪些 embedding。

---

## 1. curl 调 deepseek-chat

脚本：`code/curl_deepseek.sh`

```bash
bash code/curl_deepseek.sh "用一句话说明 RAG 是什么"
bash code/curl_deepseek.sh --raw "你好"        # 打印完整 JSON（含 usage）
bash code/curl_deepseek.sh --local "你好"      # 强制走本机 Ollama
bash code/curl_deepseek.sh --cloud "你好"      # 强制走 AIHubmix（需真实 key）
```

实际输出：

```
POST http://127.0.0.1:11434/v1/chat/completions
model=deepseek-chat  question=用一句话说明 RAG 是什么
----
RAG 是一种结合了检索和生成的上下文学习框架，用于生成高质量的回答。

[tokens] prompt=30 completion=21 total=51
```

它读的是同一份 `code/.env`，所以默认指向本机 Ollama 的 `deepseek-chat`（Ollama 里是 `qwen2.5-coder:3b` 的别名）。
换真实云端 key 只需：

```bash
LLM_API_KEY=sk-xxxx bash code/curl_deepseek.sh --cloud "你好"
```

脚本内部就是标准的 OpenAI 兼容请求，可以照抄：

```bash
curl http://127.0.0.1:11434/v1/chat/completions \
  -H "Authorization: Bearer ollama" \
  -H "Content-Type: application/json" \
  -d '{"model":"deepseek-chat","messages":[{"role":"user","content":"你好"}],"stream":false}'
```

> 没有 Key 时 `--cloud` 会打印 `HTTP 错误: {"message": "invalid key: ..."}` 并以 exit 1 退出，方便排查。

---

## 2. 怎么看向量数据库里存的数据

有两个办法：**命令行巡检**（推荐，能看清 schema/索引/向量数值）和 **Attu 网页界面**。

### 2.1 命令行：`code/inspect_vectors.py`

```bash
python3 code/inspect_vectors.py                       # 全扫：Milvus + FAISS + LlamaIndex + Chroma
python3 code/inspect_vectors.py --milvus              # 只看 Milvus 所有 collection
python3 code/inspect_vectors.py --milvus demo_hybrid_bgem3 -n 2 --head 3
python3 code/inspect_vectors.py --faiss code/C3/faiss_index_store
python3 code/inspect_vectors.py --llamaindex code/C3/llamaindex_index_store
python3 code/inspect_vectors.py --milvus demo_text_dense --dump /tmp/vec.json
```

输出的内容（`--milvus demo_hybrid_bgem3 -n 2 --head 3`）：

```
Milvus collection: demo_hybrid_bgem3
行数: 6
字段:
  - id                     INT64                 [主键]
  - text                   VARCHAR              max_length=4096
  - dense                  FLOAT_VECTOR         dim=1024
  - sparse                 SPARSE_FLOAT_VECTOR （稀疏，维度=词表大小）
索引:
  - dense:  {"index_type": "AUTOINDEX", "metric_type": "IP", ...}
  - sparse: {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP", ...}

  [1]
      id: 0
      text: 悬崖上的白龙 一头雄伟的白色巨龙栖息在悬崖边缘…
      dense: [0.0063539133, 0.010740685, -0.051702984, … (+1021 维)]  dim=1024
      sparse: {6:0.0065, 43:0.0451, 213:0.0445, … (+37 项)}  nnz=40
```

四个库都能读：

| 存储 | 位置 | 读法 |
| --- | --- | --- |
| Milvus | `localhost:19530` | `--milvus`，走 pymilvus `query(output_fields=["*"])` |
| FAISS | `code/C3/faiss_index_store/` | `--faiss`，读 `index.faiss` + `index.pkl`（向量 + 原文） |
| LlamaIndex | `code/C3/llamaindex_index_store/` | `--llamaindex`，读 `default__vector_store.json` |
| Chroma | `C4/02`、`C4/04` 用的 | 脚本没传 `persist_directory`，是内存模式，**进程退出即丢** |

### 2.2 为什么平时看 Milvus 是空的

`C3/04`、`C3/work_*`、`C4/01`、`C4/text2sql` 在结束处都调了 `drop_collection`（教程本意是「跑完清理」）。
所以跑一次教程脚本，collection 就被删一次。

想看数据，先写一份能留存的演示数据：

```bash
python3 code/demo_write_vectors.py     # 建 4 个 demo_* collection，不删
python3 code/inspect_vectors.py --milvus
python3 code/demo_write_vectors.py --clean   # 看完清掉
```

写入的 4 个 collection 正好覆盖 4 种典型编码：

| collection | 编码方式 | 维度 | 行数 |
| --- | --- | --- | --- |
| `demo_text_dense` | bge-small-zh-v1.5 | 512 稠密 | 3 |
| `demo_hybrid_bgem3` | BGE-M3 | 1024 稠密 + 稀疏(词表 250002) | 6 |
| `demo_multimodal` | Visualized-BGE | 768（图文同空间） | 7 |
| `demo_tfidf_sparse` | TF-IDF | 稀疏(词表 61) | 6 |

### 2.3 网页界面：Attu

已经加进 `code/docker-compose.local.yml`，跟着一起启动：

```bash
docker compose -f code/docker-compose.local.yml up -d
# 浏览器打开 http://localhost:18000 ，Address 填 host.docker.internal:19530
```

界面上能点开 collection → 看 schema、索引、Data 页翻数据、Vectors 页看单个向量。

### 2.4 直接查 API / Python

```bash
# Milvus 的 REST 接口
curl -s -X POST http://127.0.0.1:19530/v2/vectordb/collections/list -d '{}'
```

```python
from pymilvus import MilvusClient
c = MilvusClient(uri="http://localhost:19530")
print(c.list_collections())
print(c.describe_collection("demo_text_dense"))
rows = c.query("demo_text_dense", filter="", output_fields=["*"], limit=3)
print(len(rows[0]["vector"]))   # 512
```

---

## 3. 测试里用到 embedding 了吗？用的什么模型？

用到了，而且是整条链路的核心。C1~C4 一共 5 种编码方式，`python3 code/demo_embedding.py` 会把它们逐个跑一遍。

### 3.1 一览

| # | 模型 | 输入 | 输出 | 用在哪些脚本 |
| --- | --- | --- | --- | --- |
| 1 | `BAAI/bge-small-zh-v1.5` | 文本 | 512 维稠密，L2 归一化 | C1/01、C1/02、C2/04、C3/02、C3/03、C3/05、C3/06、C3/07、C4/02、C4/04、C4/06 |
| 2 | `BAAI/bge-small-en` | 英文文本 | 384 维 | C3/05（英文 PDF） |
| 3 | `BAAI/bge-large-zh-v1.5` | 文本 | 1024 维 | C4/07、C4/work_rerank（重排前的召回） |
| 4 | `BAAI/bge-base-en-v1.5` + `Visualized_base_en_v1.5.pth` | 图片 / 文本 / 图文 | 768 维（同一空间） | C3/01、C3/04、C3/work_multimodal、C3/work_hybrid |
| 5 | `BAAI/bge-m3` | 文本 | 1024 稠密 + 稀疏(250002 词表) | C4/01、C4/text2sql/knowledge_base |
| 6 | `google/siglip-base-patch16-256-multilingual` | 文本 | 768 维 | C4/01_v2 |
| 7 | `sklearn TfidfVectorizer` | 文本 | 稀疏（词表大小） | C4/01_v2、demo |
| 8 | `bert-base-uncased` | 文本 | `[batch, token数, 768]` token 级 | C4/07（ColBERT MaxSim 重排） |

### 3.2 文本 → 向量（最常用的一路）

```python
from langchain_huggingface import HuggingFaceEmbeddings

emb = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},   # L2 归一化，配合 COSINE 用
)
vecs = emb.embed_documents(["张三是法外狂徒", "FAISS 是一个向量检索库"])  # 建库：批量
q    = emb.embed_query("FAISS 是做什么的？")                              # 查询：单条
```

实测：`bge-small-zh-v1.5` → 512 维，范数 1.0000；`悬崖上的巨龙` vs `一只可爱的小猫` 余弦相似度 0.2313。

LlamaIndex 里的等价写法是 `Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-small-zh-v1.5")`，
索引/检索时自动调用，不需要手动 embed。

### 3.3 图片 / 图文 → 向量（多模态）

`Visualized_BGE` 是「BGE 文本塔 + EVA-CLIP 视觉塔」拼的模型，图、文、图文都输出 **768 维且在同一空间**：

```python
from visual_bge.visual_bge.modeling import Visualized_BGE

model = Visualized_BGE(model_name_bge="BAAI/bge-base-en-v1.5",
                       model_weight="../../models/bge/Visualized_base_en_v1.5.pth")
model.eval()
with torch.no_grad():
    t  = model.encode(text="datawhale开源组织的logo")          # (1, 768)
    i  = model.encode(image="../../data/C3/dragon/dragon02.png") # (1, 768)
    ti = model.encode(image="...png", text="悬崖上的白龙")        # (1, 768)
    sim = i @ ti.T                                              # 0.7799
```

因为同空间，所以 `C3/04` 能把所有图片编码入库、再用「查询图 + 查询文本」的联合向量去检索；
`demo_multimodal` collection 里存的就是这 7 张图的向量。

### 3.4 稠密 + 稀疏（混合检索）

`BGE-M3` 一次前向同时给出两种向量：

```python
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
ef = BGEM3EmbeddingFunction(use_fp16=False, device="cpu")
out = ef(["悬崖上的巨龙"])
out["dense"]    # (1, 1024)，归一化
out["sparse"]   # csr 矩阵，词表 250002 维，本条只有 6 个非零项
```

稀疏向量的下标是词表 token id、值是该 token 的权重（实测 `{6:0.0659, 7977:0.1459, 14732:0.2959, ...}`），
密度只有 0.0024%，所以 Milvus 用 `SPARSE_INVERTED_INDEX` 存，跟稠密向量做 RRF 融合（`C4/01`）。

`C4/01_v2` 则是用 SigLIP 出稠密、用 TF-IDF 手工造稀疏（`TfidfVectorizer(max_features=10000, ngram_range=(1,2))`）。

### 3.5 token 级向量（只用于重排）

`C4/07` 的 ColBERT 重排器不是「一句一向量」，而是每个 token 一个 768 维向量：

```python
inputs = tokenizer([query], return_tensors="pt", padding=True, truncation=True, max_length=128)
hidden = model(**inputs).last_hidden_state          # (1, 7, 768)
hidden = F.normalize(hidden, p=2, dim=-1)
# 每个 query token 对 doc 所有 token 取 max 相似度，再求和 → MaxSim 分数
```

---

## 4. 一句话总结

- **curl 脚本**：`code/curl_deepseek.sh`，默认读 `.env` 走本地 Ollama，`--cloud` 切云端。
- **看数据**：`python3 code/inspect_vectors.py --milvus`（命令行）或 `http://localhost:18000`（Attu 界面）；
  教程脚本会 drop collection，先跑 `python3 code/demo_write_vectors.py` 造出可留存的数据。
- **embedding**：用了 8 种，核心是 `bge-small-zh-v1.5`（512 维文本）、`Visualized-BGE`（768 维图文同空间）、
  `BGE-M3`（1024 稠密 + 稀疏）、`TF-IDF`/`SigLIP`（C4/01_v2）、`bert-base-uncased`（ColBERT 重排）。
  `python3 code/demo_embedding.py` 可一次性看到每种模型的维度、范数和相似度数值。
