# C4 导读：检索优化 —— 怎么自己跑、看什么

> 配套文档：`docs/chapter4/`（混合检索、元数据过滤、Text2SQL、查询路由、重排精炼）
> 配套脚本：`code/C4/01_hybrid_search.py` ~ `code/C4/work_rerank_and_refine.py`

---

## 一、十个脚本怎么跑

C4 按依赖分三组：

| 组 | 脚本 | 需要什么 |
| --- | --- | --- |
| **混合检索** | 01 `_v2` | **Milvus**（Docker）+ BGE-M3 / SigLIP 模型 |
| **元数据 & SQL** | 02 03 `_v2` 04 | 本地 LLM（Ollama）；（02/04 还要能访问 B 站） |
| **路由 & 重排** | 05 06 07 `work_*` | 本地 LLM（Ollama） |

前置条件：

```bash
# Milvus（01 / 01_v2 需要）
docker compose -f code/docker-compose.local.yml up -d

# 本地 LLM（02/03/04/05/06/07 需要）
ollama serve &                       # 或已在后台运行
ollama list                          # 确认 deepseek-chat 之类模型在
```

然后从仓库任意位置跑：

```bash
python3 code/C4/01_hybrid_search.py              # ~30 s
python3 code/C4/01_hybrid_search_v2.py           # ~14 s
python3 code/C4/02_text_to_metadata_filter.py    # ~49 s
python3 code/C4/03_text2sql_demo.py              # ~1 min 52 s
python3 code/C4/03_text2sql_demo_v2.py           # ~6 s
python3 code/C4/04_text_to_metadata_filter_v2.py # ~35 s
python3 code/C4/05_llm_based_routing.py          # ~29 s
python3 code/C4/06_embedding_based_routing.py    # ~1 min
python3 code/C4/07_rerank_and_refine.py          # ~24 s
python3 code/C4/work_rerank_and_refine.py        # ~20 s
```

关掉分阶段日志：`RAG_LOG=quiet python3 code/C4/01_hybrid_search.py`
一次跑完整章：`bash code/run_all.sh C4`

---

## 二、C4 讲什么、要教会你什么

C3 结束时，你有了能用的索引和几种检索策略。C4 问的是：

> **索引建好之后，怎么把「能用」调到「好用」？**

四个递进的主题：

| 小节 | 脚本 | 教会你什么 |
| --- | --- | --- |
| 混合检索 | 01 `_v2` | 一路向量有盲区，**两路互补 + RRF 融合**才稳 |
| 元数据 & SQL | 02 03 04 | 不是所有问题都该走向量；**结构化条件/SQL** 往往更对 |
| 查询路由 | 05 06 | **先判断问题类型，再选链路**——路由器是 RAG 的分诊台 |
| 重排精炼 | 07 `work` | 粗排不准就**重排**，太啰嗦就**压缩** |

---

## 三、01 / 01_v2 · 混合检索

### 01：BGE-M3 出双路向量

```bash
python3 code/C4/01_hybrid_search.py
```

查询编码这一步最关键：

```
稠密向量：dim=1024，范数=1.0000
稀疏向量：dim=250002，非零 6 个
稀疏密度约 0.002400% —— 绝大多数位置是 0，所以用倒排存
```

**要学到的东西：**

1. **稀疏向量密度极低**（6 / 25 万个位置非零，约万分之一）。
   所以必须用 `SPARSE_INVERTED_INDEX`（倒排表），不能用普通向量索引——否则内存直接爆。
2. BGE-M3 的好处是**一次前向同时产出 dense 和 sparse**，不用维护两套模型。
3. 两路的排序和分数完全不同：

   | 名次 | 稠密（IP 分数） | 稀疏（IP 分数） |
   | --- | --- | --- |
   | 1 | 悬崖上的白龙 0.7214 | 悬崖上的白龙 0.2254 |
   | 2 | 中华金龙 0.5353 | 中华金龙 0.0857 |
   | 3 | 驯龙高手：无牙仔 0.5231 | 驯龙高手：无牙仔 0.0639 |

   注意：**两路的分数绝对值完全不可比**（0.72 vs 0.23），所以不能简单加权求和。

4. 这就是为什么要 **RRF**：它只看名次不看分数，`score = Σ 1/(k + rank)`，
   两路分数量纲不同也能直接融合。**RRF 的分数不是相似度，别拿去和 IP 分数比大小。**
5. `expr` 过滤器在检索前生效，能先砍掉大半候选。

耗时分布（实测 ~30 s）：

```
1. 连接 Milvus + 加载 BGE-M3    6.25 s
2. 建 Collection + 两套索引     1.22 s
3. 生成双向量并入库            14.55 s   ← 最慢
4. 查询编码                     5.95 s
5. 单路检索对比                 625 ms
6. RRF 混合检索 + 清理          1.12 s
```

### 01_v2：SigLIP + TF-IDF

和 01 的结构**完全一样**，唯一差别是两路向量的来源：

| | 稀疏来自哪 | 性质 |
| --- | --- | --- |
| 01 | BGE-M3 学出来的词权重 | 需要训练，懂同义 |
| 01_v2 | TF-IDF 统计出来的权重 | 零训练、可解释、换语料直接能用 |

**两种方案的 Collection schema 完全一样**——说明换编码器不影响存储结构，
这是设计向量库时该记住的一点。01_v2 快很多（~14 s vs ~30 s），因为 TF-IDF 是纯 CPU 统计。

---

## 四、02 / 04 · 自然语言 → 元数据过滤

### 02：SelfQueryRetriever

```bash
python3 code/C4/02_text_to_metadata_filter.py
```

做法：**先一次 LLM 调用，把自然语言拆成两部分**——

```
语义 query     → 真正拿去算相似度的文本
metadata filter → 交给向量库的结构化条件
```

日志里会打印它生成的结构化查询：

```
Generated Query: query='600' filter=Comparison(
    comparator=Comparator.GT: 'gt', attribute='length', value=600) limit=10
```

**要学到的东西：**

1. 「时间最短的视频」→ 生成的是**排序/过滤条件**，不是靠向量相似度猜。
2. **元数据字段的 type 必须写对**：`length` 若声明成 `string`，`> 600` 就无法比较。
3. LLM 只能按 `metadata_field_info` 里**声明过的字段**生成条件，没声明的不可能用。
4. 代价：每次查询多一次 LLM 调用，字段多时 prompt 会变长。

耗时（~49 s）：建 Chroma 22.5 s / 执行查询 23.6 s，其余接近 0。

### 04：自己出 JSON 指令，代码里排序

```bash
python3 code/C4/04_text_to_metadata_filter_v2.py
```

实测输出：

```
--- 原始查询: '时间最短的视频' ---
  生成指令：{'sort_by': 'length', 'order': 'asc'}
  Top1：《吴恩达 x OpenAI Prompt课程》【专业翻译  390s  54133 次

--- 原始查询: '播放量最高的视频' ---
  生成指令：{'sort_by': 'view_count', 'order': 'desc'}
  Top1：《吴恩达 x OpenAI Prompt课程》【专业翻译  390s  54133 次
```

**和 02 的差别：**

| | 02 (SelfQuery) | 04_v2 (自写 prompt) |
| --- | --- | --- |
| LLM 输出 | 结构化 filter | `{"sort_by":..., "order":...}` |
| 谁执行 | 向量库 | Python `sorted()` |
| 灵活性 | 受向量库算子限制 | 任意复杂逻辑都能写 |
| 成本 | 框架帮你写 prompt | prompt 自己维护，字段多了容易失控 |

**关键技巧：`response_format={"type": "json_object"}`**。
不加这个，LLM 常会加解释文字，`json.loads` 直接崩。

**另一个关键点**：拿到 LLM 的 JSON 后**必须校验白名单**——
`sort_by` 只能是 `length`/`view_count`，`order` 只能是 `asc`/`desc`。
否则 LLM 返回非法字段时，排序会静默出错。

---

## 五、03 · Text2SQL

### 03：schema → 现场生成 SQL

```bash
python3 code/C4/03_text2sql_demo.py
```

sqlite 造了个电商库（users 5 行 / products 7 行 / orders 7 行，含两个外键）。
6 个问题实测 6/6 成功：

```
问题 1: 查询所有用户的姓名和邮箱
  SQL：SELECT name, email FROM users                              → 5 行
问题 2: 年龄大于30的用户有哪些
  SQL：SELECT name, email FROM users WHERE age > 30;              → 2 行
问题 4: 查询来自北京的用户的所有订单
  SQL：SELECT o.*, u.name as user_name FROM orders o
       JOIN users u ON o.user_id = u.id WHERE u.city = '北京'     → 2 行
问题 5: 统计每个城市的用户数量
  SQL：SELECT city, COUNT(*) as count FROM users GROUP BY city    → 5 行
```

**要学到的东西：**

1. **不是所有问题都该走向量检索。** 「统计每个城市的用户数量」的正确答案是
   `GROUP BY`，向量相似度给不出来。
2. agent 会**先读 schema 再生成 SQL**——schema 描述越全，SQL 越准。
3. 跨表 JOIN 也能生成（问题 4），因为外键关系在 schema 里可见。

⚠️ **生产必做的三件事**（脚本里没做，但你必须知道）：
- 只允许 `SELECT`，禁止 `DROP`/`DELETE`/`UPDATE`；
- 加 `LIMIT` 和执行超时；
- 用只读账号连库，防止慢查询打挂库。

耗时（~1 min 52 s）：初始化 56 s + 6 个查询 56 s，就是 12 次 LLM 调用。

### 03_v2：示例检索驱动

```bash
python3 code/C4/03_text2sql_demo_v2.py
```

思路换了个方向：**先把「问题→SQL」的历史示例存进向量库，检索相似示例再拼 prompt**（few-shot 检索）。

和 03 的差别：

- 03 靠 **schema 描述**约束写法；
- 03_v2 靠 **相似示例**约束写法（对字段名、聚合口径、日期格式更直接）。

这个 demo 本身**不调 LLM**（只做编码 + 执行 SQL），所以 Ollama 没开也能跑通（~6 s）。

> 示例库要持续维护：业务新增表，就得补新的「问题→SQL」对。

---

## 六、05 / 06 · 查询路由

两个脚本解决同一个问题：**用户问什么，就该走哪条链路？**

### 05：LLM 分类路由

```bash
python3 code/C4/05_llm_based_routing.py
```

用 `RunnableBranch` 按顺序匹配，第一个为 True 的分支胜出：

```python
RunnableBranch(
    (lambda x: "川菜" in x["topic"], sichuan_chain),
    (lambda x: "粤菜" in x["topic"], cantonese_chain),
    general_chain,   # 默认兜底
)
```

实测结果——**注意这里翻车了，而且是很好的教学素材**：

```
问题 1: 麻婆豆腐怎么做？          → 路由决策：川菜   ✓ 符合预期
问题 2: 白切鸡的正宗做法是什么？  → 路由决策：川菜   ✗ 应该是粤菜
问题 3: 番茄炒蛋需要放糖吗？      → 路由决策：粤菜   ✗ 应该是其他
```

**要学到的东西：**

1. **路由靠 LLM 分类，不是关键词匹配**——所以口径不统一时就会误判。
   白切鸡是粤菜，但分类器看到「正宗做法」就往川菜那边靠了。
2. 分类器温度设 0 是为了**同一问题每次分级稳定**，但不代表分得对。
3. 成本：**一次分类 + 一次回答 = 2 次 LLM 调用**，比不路由翻倍。
4. 分类失败（返回不在列表里的词）不会报错，会**静默落到通用链**——
   这既是兜底，也是隐患（错了你也不会知道）。

生产里的改进：把分类结果缓存、用更小的模型做路由、或者**先用规则粗筛**。

### 06：嵌入相似度路由

```bash
python3 code/C4/06_embedding_based_routing.py
```

不调 LLM 分类，而是**把「菜系描述」编码成向量，用余弦相似度选路由**：

```
问题 1: 水煮鱼怎么做才嫩？      川菜=0.5022  粤菜=0.4730
问题 2: 如何做一碗清淡的云吞面？ 川菜=0.4688  粤菜=0.5747
问题 3: 麻婆豆腐的核心调料是什么？ 川菜=0.5903  粤菜=0.4149
```

**要学到的东西：**

1. **便宜、快、稳定**：0 次 LLM 分类调用，只用嵌入模型算一次相似度。
2. 但**只认描述里写过的词**——路由描述里没提「云吞面」，问题提到它就可能分错。
3. **分数接近时不可靠**：问题 1 是 0.5022 vs 0.4730，差距只有 0.03，
   这种 borderline case 换个模型可能就翻盘了。**生产里要设阈值或回退到通用链。**
4. 路由描述怎么写得看业务：把用户真实问法里的高频词放进去，命中率才高。

**05 vs 06 对比：**

| | 05 (LLM 分类) | 06 (嵌入相似度) |
| --- | --- | --- |
| LLM 调用 | 2 次 | 0 次（分类层面） |
| 准确度 | 能理解语义，但会误判 | 只认描述里的词 |
| 成本 | 高 | 低 |
| 稳定性 | 温度 0 仍可能有波动 | 确定性（同一输入同一结果） |

---

## 七、07 / work · 重排 + 精炼

### 07：ColBERT 重排 → LLM 压缩

```bash
python3 code/C4/07_rerank_and_refine.py
```

三个环节：

```
粗排（向量，快）  → 5 条
重排（ColBERT，准）→ 重新排序
精炼（LLM，省）    → 每条只剩相关句子，甚至整条丢掉
```

实测：**粗排 5 条 → 精炼后 1 条**（5→1）。

那 4 条被丢掉的，就是「噪音」——粗排召回但实际不相关的。

**要学到的东西：**

1. **ColBERT 只能重排不能召回**：它要给每对 `(query, doc)` 算逐 token 交互，
   全库排序太贵；所以标准做法是「粗排大 K → 重排小 K」。
2. `k` 设大一点（20）是给重排留足候选——**粗排漏掉的，重排也救不回来**。
3. **LLM 压缩的收益是省 token，代价是每条文档一次 LLM 调用**——
   20 条就是 20 次。生产里常换成更便宜的模型，或只对 Top-5 压缩。
4. ⚠️ `LLMChainExtractor` 自带的提示词是**英文**的，中文场景下模型常直接返回空串。
   脚本里换了中文 prompt，并且过滤掉「无」这类占位输出
   （基类只判断 `len(output) == 0`，而中文返回「无」长度不为 0，会被误当成结果）。

耗时（~24 s）：粗排 4.4 s + 重排精炼 12.2 s。

### work：中文友好分块

```bash
python3 code/C4/work_rerank_and_refine.py
```

和 07 **唯一的变量是分块策略**：

| | chunk_size | overlap | 分隔符 |
| --- | --- | --- | --- |
| 07 | 500 | 100 | 默认（英文） |
| work | 300 | 20 | 中文标点优先 `。！？；，` |

实测：**粗排 9 条 → 精炼后 1 条**，精炼后总字符数只有 41。

**要学到的东西：**

1. 中文标点优先能让一个 chunk 尽量停在句末，**语义更完整**。
2. chunk 小 → 召回多、噪音也多；chunk 大 → 上下文全、token 贵。
3. 判断分块好坏没有标准答案，要看下游是「精确问答」还是「摘要归纳」。
4. overlap 从 100 降到 20 是为了减少相邻 chunk 的重复内容。

---

## 八、自己动手验证（建议按顺序做）

### 1. 亲眼看到「两路分数不可比」

跑 `01`，把稠密和稀疏的 Top5 分数抄下来。你会发现稠密都是 0.5~0.7，
稀疏都是 0.06~0.23。**这就是为什么不能用加权求和，只能用 RRF。**

### 2. 复现路由翻车

`05` 里问题 2（白切鸡）被分到川菜。改分类器的 prompt，明确写出
「白切鸡、烧鹅、虾饺 → 粤菜」，看能不能修好。

或者跑 `06`，看问题 1 的 0.5022 vs 0.4730 —— 这种微小差距，
换个嵌入模型（比如换成 bge-m3）结果可能就反过来了。

### 3. 改 `k` 看重排的效果

`07` 里把 `TOPK` 从 20 改成 3 和 50：
- k=3 → 重排几乎没用（候选太少）
- k=50 → 重排有更多发挥空间，但 LLM 压缩的成本翻倍

### 4. 对比两种 Text2SQL 思路

跑 `03`（schema 驱动）和 `03_v2`（示例驱动），问同一个复杂问题
（比如「哪个用户买的最多」），看两者生成的 SQL 差在哪。

---

## 九、常见问题

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| `Connection refused` / `localhost:19530` | Milvus 没起 | `docker compose -f code/docker-compose.local.yml up -d` |
| `openai.AuthenticationError: 401` | 没配 `code/.env` | 参考 `code/RUN_NOTES.md` 建 `.env` 指向 Ollama |
| `加载 BiliBili 视频失败` / 没有成功加载任何视频 | 网络不通或 B 站接口变了 | 检查网络；02/04 依赖 B 站接口 |
| 路由结果和预期不符 | LLM 分类误判（正常现象） | 改 prompt 或换路由描述，见观察点 |
| 07 压缩后一条不剩 | 用了英文默认 prompt | 脚本已换成中文 prompt，别改回去 |
| `json.loads` 失败 | LLM 输出了多余文字 | 确认 `response_format=json_object` 还在 |
| 想关掉阶段日志 | — | `RAG_LOG=quiet python3 ...` |

---

## 十、C4 和前后章节的关系

```
C3  索引构建：块/句子/表格 → 向量 → FAISS / Milvus（含 HNSW）
C4  检索优化 ← 你在这一章
     01/01v2 混合检索：单路有盲区，RRF 融合
     02/04   元数据过滤：结构化条件 + 语义
     03       Text2SQL：该算数就交给数据库
     05/06    查询路由：先分诊，再选链路
     07/work  重排精炼：粗排后精排，再压缩省 token
```

**一句话总结 C4**：索引建好只是起点。真正决定效果的是——
**知道哪类问题该走哪条路**（路由/元数据/SQL），
**以及怎么把召回结果从「差不多」变「刚刚好」**（混合检索 + 重排 + 精炼）。
