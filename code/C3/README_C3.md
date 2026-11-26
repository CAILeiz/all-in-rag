# C3 导读：索引构建 —— 怎么自己跑、看什么

> 配套文档：`docs/chapter3/`（向量数据库、多模态检索、句子窗口、递归检索）
> 配套脚本：`code/C3/01_bge_visualized.py` ~ `code/C3/work_hybrid_multimodal_search.py`

---

## 一、十个脚本怎么跑

C3 分两组，依赖不同：

| 组 | 脚本 | 需要什么 |
| --- | --- | --- |
| **基础组** | 01 02 03 | 只要能联网下模型（首次约 375 MB） |
| **多模态组** | 04 work_* | **还要 Milvus**（Docker）+ Visualized BGE 权重 |
| **检索策略组** | 05 06 07 | 要本地 LLM（Ollama）+ Excel/PDF 数据 |

先确认前置条件：

```bash
# 1. 下 Visualized BGE 权重（375 MB，只需一次；已存在会自动跳过）
python3 code/C3/download_model.py

# 2. 起 Milvus（04 / work_* 需要）
docker compose -f code/docker-compose.local.yml up -d
docker compose -f code/docker-compose.local.yml ps   # 4 个容器都该是 healthy
```

然后从仓库任意位置跑（脚本会自动锚定工作目录）：

```bash
python3 code/C3/01_bge_visualized.py            # ~9 s
python3 code/C3/02_langchain_faiss.py           # ~18 s
python3 code/C3/03_llamaindex_vector.py         # ~11 s
python3 code/C3/04_multi_milvus.py              # ~14 s
python3 code/C3/05_sentence_window_retrieval.py # ~2 min 23 s（最慢）
python3 code/C3/06_recursive_retrieval.py       # ~16 s
python3 code/C3/07_recursive_retrieval_v2.py    # ~18 s
python3 code/C3/work_multimodal_dragon_search.py    # ~10 s
python3 code/C3/work_hybrid_multimodal_search.py    # ~23 s
```

关掉分阶段日志，只看原始输出：

```bash
RAG_LOG=quiet python3 code/C3/02_langchain_faiss.py
```

一次跑完整章并汇总：

```bash
bash code/run_all.sh C3
# 日志落在 code/logs/C3/<脚本名>.log，汇总在 code/logs/summary.txt
```

---

## 二、C3 讲什么、要教会你什么

C2 结束时你手上有一堆**切好的文本块**。C3 回答下一个问题：

> **这些块怎么变成向量、存到哪里、怎么按不同策略取回来？**

三个递进的主题：

| 小节 | 脚本 | 教会你什么 |
| --- | --- | --- |
| 向量库基础 | 01 02 03 | 三种姿势建索引：直接编码 / LangChain / LlamaIndex；索引要能**落盘** |
| 多模态检索 | 04 work_* | 图和文可以编进**同一个向量空间**；单路不够就上混合检索 |
| 检索策略 | 05 06 07 | 检索单元不一定是「块」：可以是一句话（窗口）、也可以是一张表（递归） |

---

## 三、01 · Visualized BGE：图和文在同一个空间

```bash
python3 code/C3/01_bge_visualized.py
```

关键输出：

```
[1/3] ✓ 完成  5.61 s  | 已加载
[2/3] ✓ 完成  957 ms  | (1, 768)
[3/3] ✓ 完成  0 ms  | 4 组相似度
```

**要学到的东西：**

1. 三种入参——**纯图像**、**纯文本**、**图像+文本**——返回的向量维度都是 `(1, 768)`，
   所以可以直接点积比较，这是多模态检索能做的基础。
2. 实测四组相似度：

   | 对比 | 相似度 |
   | --- | --- |
   | 纯图像 vs 纯图像 | 0.8318 |
   | 图文结合1 vs 纯图像 | 0.8291 |
   | 图文结合1 vs 纯文本 | 0.7627 |
   | 图文结合1 vs 图文结合2 | **0.9058** |

3. **图文结合的两个图像最像（0.9058）**——因为两边都同时带视觉和语义信号，
   比单看图像或单看文本都更稳。这就是后面 `work_hybrid_multimodal_search`
   里 MULTIMODAL 模式的底气。
4. 模型已做归一化，点积等价于余弦相似度，不需要手动 normalize。

---

## 四、02/03 · 建索引的两种框架写法

两个脚本干的是同一件事（3 条文档 → 向量 → 落盘 → 重载 → 检索），
但框架不同，代码风格差别很大。

### 02：LangChain + FAISS

```
[1/4] ✓ 完成  0 ms  | 3 条
[2/4] ✓ 完成  10.89 s  | 已加载
[3/4] ✓ 完成  4.61 s  | 已落盘
[4/4] ✓ 完成  2.71 s  | Top1 命中
       命中：FAISS是一个用于高效相似性搜索和密集向量聚类的库。
```

要点：

1. `FAISS.from_documents(docs, embeddings)` 一步建索引，**embeddings 是显式传的**。
2. 落盘是 `save_local("./faiss_index_store")` → 生成 `index.faiss` + `index.pkl` 两个文件。
3. 第 4 步特意**重新从磁盘 load**，验证索引真的持久化了，不是只活在内存里。

### 03：LlamaIndex + Settings

```
[1/3] ✓ 完成  6.79 s  | 已配置
[2/3] ✓ 完成  4.09 s  | 已建索引
[3/3] ✓ 完成  8 ms  | 已落盘
```

要点：

1. 只配一次 `Settings.embed_model`，之后 `from_documents` 自动用它——**代码更短**。
2. 落盘是 `storage_context.persist()` → 一整套 JSON（`docstore.json` /
   `index_store.json` / `vector_store.json`），比 FAISS 的两个二进制文件更透明、可读。
3. LlamaIndex 默认还会做 node 切分和元数据维护，这是为 05/06/07 的检索策略铺路。

**两者对比：**

| | 02 (FAISS) | 03 (LlamaIndex) |
| --- | --- | --- |
| 建索引 | 显式传 embeddings | Settings 全局配置 |
| 落盘格式 | 2 个二进制文件 | 多个 JSON |
| 强项 | 极简、快 | 后续检索策略丰富 |

> 两者底层可以互通：LlamaIndex 也能把 FAISS / Milvus 当作 vector store 用。

---

## 五、04 · Milvus 多模态检索

```bash
python3 code/C3/04_multi_milvus.py
```

七个阶段（节选）：

```
[1/7] ✓ 完成  5.82 s  | 已连接
[2/7] ✓ 完成  264 ms  | dim=768
[3/7] ✓ 完成  1.22 s  | 7 条向量
[4/7] ✓ 完成  24 ms   | 7 条
[5/7] ▶ 建 HNSW 索引并加载
       index_type  HNSW
       metric      COSINE
       params      M=16, efConstruction=256
```

**要学到的东西：**

1. Milvus 是**服务端型**向量库——连接失败会在这里直接报错，不像 FAISS 是纯进程内。
2. `dim=768` 必须和编码器输出严格对上，否则插入就失败。
3. **不建索引也能搜，但数据量一大就退化成暴力扫描。** HNSW 是图结构近邻索引，
   `M` 控制每个节点的连边数（越大越准越占内存），`efConstruction` 控制建索引时的搜索宽度。
4. 检索时还有个 `ef` 参数（查询期搜索宽度，脚本里是 128），**建索引和查询是两套参数**。

实测 Top1 命中 `dragon/query.png`：
- Top1 distance **0.9466**（COSINE，越接近 1 越像）
- Top5 最低 **0.5360** —— 名次掉一点，分数掉得很快，说明排序有区分度。

---

## 六、05/06/07 · 检索单元不一定是「块」

这三个脚本是 C3 最"反直觉"的部分：它们都在挑战
**「一个向量对应一个固定大小的文本块」**这个默认假设。

### 05：句子窗口检索（Sentence Window）

```
1. 配置 LLM 与嵌入模型          7.70 s
2. 加载 PDF                    21.11 s
3. 构建两个索引              1 min 49 s   ← 最慢的一步
4. 组装查询引擎                   1 ms
5. 两种检索对比                 5.52 s
```

做法：**按单个句子建向量索引**（定位极准），但检索到某句后，
不是返回那一句，而是返回它前后 `window_size=3` 句组成的窗口（上下文够用）。

- 好处：既能精准定位「答案落在哪一句」，又能给 LLM 足够上下文。
- 代价：节点数暴增（每句一个），索引体积和检索开销都上升。
- 和常规分块的差别：常规是「块切小 + overlap 兜底」，句子窗口是
  「块切极小 + 上下文外挂」。**同一份句子可以配不同窗口大小，不用重建向量。**

### 06：递归检索（Pandas 引擎）

```
1. 配置模型                     6.91 s
2. 为每个工作表建查询引擎 + 摘要节点   181 ms
3. 用摘要节点建顶层向量索引     5.03 s
4. 组装 RecursiveRetriever        1 ms
5. 执行查询                     3.89 s
```

两级结构：**先按年份路由到对应工作表，再在那个表里查**。

日志里能看到两次 `Retrieving`：第一次 `query id = None`（顶层摘要索引），
命中后跳到 `年份_1994`。

它生成的是 **pandas 代码**，不是文本：

```python
df[df['评分人数'] == df['评分人数'].min()]['电影名称'].iloc[0]
```

**这正是向量检索做不到的**：相似度找得到「相关段落」，但算不出「最小值」。

- 设计关键：`index_id` 必须和 `query_engine_dict` 的 key 完全一致，否则跳不过去。
- 实测答案：`燃情岁月`。
- ⚠️ `PandasQueryEngine` 会**把问题转成代码执行**，务必只对可信数据启用。

### 07：递归检索 v2（先过滤再检索）

```
1. 配置模型                     7.33 s
2. 加载 Excel 并生成两类文档    182 ms
3. 建摘要索引                   4.79 s
4. 建内容索引                   2.07 s
5. 两步查询：路由 → 过滤检索    2.46 s
6. 生成回答                     1.15 s
```

和 06 的两点差别：

1. **第二层不做代码执行**：07 只是把对应表格检索出来交给 LLM 回答，更安全。
2. 用 `MetadataFilters` + `ExactMatchFilter` 做结构化过滤，
   这是「**先按条件筛，再算相似度**」的范式——生产里极常用（按租户/时间/版本过滤）。

实测答案：`《活着》410099人`。

> **注意 06 和 07 的答案不一样**（燃情岁月 vs 活着）。这不是 bug，是很好的教学点：
> 06 让 Pandas 精确算，07 靠 LLM 自己数。**算术类问题，LLM 发挥不可靠**，
> 这就是为什么 07 的观察点里写着「数字对不上就换 06 的 Pandas 引擎」。

---

## 七、work 系脚本：多模态检索的四种模式

### work_multimodal_dragon_search.py（8 步 / ~10 s）

图像入库 → HNSW → 三种查询方式对比：

- **图+文**通常最准：视觉和语义两个信号投票。
- **纯文本**依赖向量里学到的图文对应关系，描述越具体越准。
- **纯图像**最容易受背景干扰（比如都拍到了天空/悬崖）。

### work_hybrid_multimodal_search.py（6 步 / ~23 s）

四种模式对比：**MULTIMODAL**（图+文融合）/ **DENSE**（纯稠密）/ **SPARSE**（BM25 式关键词）
/ **HYBRID**（RRF 融合）。

RRF 的关键：**只看名次，不看分数绝对值**，所以两路分数量纲不同也能直接融合。
生产里 HYBRID 通常是默认选择。

---

## 八、自己动手验证（建议按顺序做）

### 1. 验证「图文结合为什么更稳」

改 `01` 里的图片/文本组合，多做几组交叉对比。你会发现：
**同源的两条图文向量相似度总是高于跨模态**，因为两个信号同向加分。

### 2. 改 HNSW 参数看召回变化

`04` 里把 `ef`（查询期参数）从 128 改成 16，再改成 256：

- 改小 → 更快，但 Top5 里可能混进不相关的图
- 改大 → 更准，但查询变慢

这就是 HNSW「用速度换准确率」的具体手感。

### 3. 改句子窗口大小

`05` 里把 `window_size` 从 3 改成 1 和 5：

- 1 → 只返回一句话，上下文不够，LLM 容易答不全
- 5 → 上下文够了，但噪音变多，还可能超出上下文窗口

### 4. 亲手复现「06 和 07 答案不一致」

连续跑两次 `06` 和 `07`，记下答案。你会发现 07 的答案偶尔会变，
而 06 的 Pandas 结果稳定——这就是「代码执行 vs 模型算术」的差别。

---

## 九、常见问题

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| `FileNotFoundError: ...Visualized_base_en_v1.5.pth` | 没下模型权重 | `python3 code/C3/download_model.py` |
| `Connection refused` / `localhost:19530` 连不上 | Milvus 没起 | `docker compose -f code/docker-compose.local.yml up -d` |
| Milvus 容器 `Exited (134)`，日志有 `precheck chunk manager client failed` | MinIO 数据目录丢了 | `mkdir -p code/volumes/minio` 后重新 `up -d` |
| `ModuleNotFoundError: No module named 'visual_bge'` | 没装本地包 | 用 `PYTHONPATH=code/C3` 跑，或 `pip install -e code/C3` |
| `openai.AuthenticationError: 401` | 没配 `code/.env` | 参考 `code/RUN_NOTES.md` 建 `.env` 指向 Ollama |
| `FontBBox ...` 刷屏 | pdfminer 的无害警告 | 忽略，或用 `grep -v FontBBox` 过滤 |
| 05 跑得特别慢（2 分钟+） | 要加载 PDF + 建两个索引 | 正常，实测 2 min 23 s |
| 想关掉阶段日志 | — | `RAG_LOG=quiet python3 ...` |

---

## 十、C3 和前后章节的关系

```
C2  数据准备：文档 → 带类型的元素 → 切好的块
C3  索引构建 ← 你在这一章
     01 多模态编码：图和文同空间
     02/03 建索引并落盘（两种框架风格）
     04 Milvus + HNSW：服务端向量库
     05 句子窗口：检索单元可以是一句话
     06/07 递归检索：检索单元可以是一张表
C4  检索优化：混合检索 / 路由 / 重排，都是在 C3 建好的索引之上做精排
```

**一句话总结 C3**：建索引不只是「把文本变成向量存起来」——
**选什么做检索单元**（块、句子、表格）、**用什么索引结构**（FLAT/HNSW/倒排）、
**存到哪里**（FAISS 本地 / Milvus 服务端），每一步都在为后面的检索质量定上限。
