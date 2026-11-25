# C2 导读：数据准备 —— 怎么自己跑、看什么

> 配套文档：`docs/chapter2/04_data_load.md`（数据加载）、`05_text_chunking.md`（文本分块）
> 配套脚本：`code/C2/01_unstructured_example.py` ~ `04_semantic_chunker.py`

---

## 一、四个脚本怎么跑

**不需要 Docker、不需要 Milvus、不需要 API Key。** 只有 04 要加载嵌入模型（约 15 秒）。

```bash
# 方式一：从仓库任意位置（脚本会自动锚定工作目录）
python3 code/C2/01_unstructured_example.py
python3 code/C2/02_character_splitter.py
python3 code/C2/03_recursive_character_splitter.py
python3 code/C2/04_semantic_chunker.py

# 方式二：先进目录，和教程文档一致
cd code/C2 && python3 02_character_splitter.py
```

关掉分阶段日志，只看原始输出：

```bash
RAG_LOG=quiet python3 code/C2/02_character_splitter.py
```

一次跑完整章并汇总：

```bash
bash code/run_all.sh C2
# PASS  C2/01_unstructured_example.py      11s
# PASS  C2/02_character_splitter.py         1s
# PASS  C2/03_recursive_character_splitter.py  0s
# PASS  C2/04_semantic_chunker.py          15s
```

---

## 二、C2 讲什么、要教会你什么

C1 里「加载 → 分块」只是两行代码。C2 把它们拆开问：

> **一份文档进 RAG 之前，到底应该长成什么样？**

答案是 C2 的两节：

| 小节 | 脚本 | 教会你什么 |
| --- | --- | --- |
| 数据加载 | 01 | 文档不是「一坨文本」，是**带类型的元素**；解析质量决定后面一切 |
| 文本分块 | 02 03 04 | 分块有 3 种思路，**块大小直接影响检索精度**（C1 埋的伏笔） |

---

## 三、01 · 数据加载：PDF 变成了什么

```bash
python3 code/C2/01_unstructured_example.py
```

关键输出：

```
[1/3] ✓ 完成  11.20 s  | 279 个元素
[2/3] ✓ 完成  0 ms  | 6 种类型
       Title                 195 个
       UncategorizedText      41 个
       Header                 22 个
       Footer                 15 个
       NarrativeText           3 个
       ListItem                3 个
```

**要学到的东西：**

1. `partition()` 返回的是 `List[Element]`，每个元素带 `category`，不是字符串。
2. 279 个元素里 **Title 占 195 个**——这份 PDF 来自网页截图，标题被切得很碎。
3. **NarrativeText 只有 3 个**——真正的正文段落极少，大量内容落在 `UncategorizedText`。
4. 所以：**解析出来的元素不能直接当分块单位用**，还得清洗（合并短元素、丢 Footer、按 Title 聚章节）。

> 报 `Is poppler installed and in PATH` 就执行 `brew install poppler`。

---

## 四、02/03/04 · 分块的三种思路

三个脚本用**同一份文件**（`data/C2/txt/蜂医.txt`，2343 字符），参数也一样（200/10），
唯一区别是**切法**。跑完对照看：

| 脚本 | 策略 | 块数 | 长度范围 | 超限块 |
| --- | --- | --- | --- | --- |
| 02 | 固定大小 `CharacterTextSplitter` | 14 | 10 ~ 294 | **3 块 >200** |
| 03 | 递归字符 `RecursiveCharacterTextSplitter` | 22 | 3 ~ 200 | 0 |
| 04 | 语义 `SemanticChunker` | **2** | 566 ~ 1776 | 不适用（没有 size 参数） |

### 02：固定大小分块，为什么「保证不了大小」

```python
CharacterTextSplitter(chunk_size=200, chunk_overlap=10)
```

它会打印警告：

```
Created a chunk of size 201, which is longer than the specified 200
Created a chunk of size 277, which is longer than the specified 200
Created a chunk of size 296, which is longer than the specified 200
```

**原因**：它只在分隔符（默认 `"\n\n"`）处切。如果某个段落本身就有 294 个字符，
它既不能把段落劈开，也不肯丢弃，只能原样保留 → **`chunk_size` 只是目标值，不是硬上限**。

另外注意块 1 只有 72 字符——短块会让向量「信息量不足」，检索时容易误命中。

### 03：递归字符分块，按层级退让

```python
RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", "。", "，", " ", ""],
    chunk_size=200, chunk_overlap=10
)
```

切分顺序：**整段 → 整行 → 句号 → 逗号 → 空格 → 单字符**。
某段太长就退一级再试，直到塞得下。结果：

```
块数：22，长度 3~200，平均 106
✓ 没有超限块（对比 02 的 3 块超限）
```

**代价**：会从句子中间断。你会看到以「。」开头、只有 3 个字符的碎片块（块 3 就是）。

> 这就是中文分块特有的坑：英文按 `" "` 断词天然安全，中文没有空格，
> 必须手动把 `"。"` `"，"` 加进 `separators`，否则整段中文根本切不开。

### 04：语义分块，按「话题变没变」切

```python
SemanticChunker(embeddings, breakpoint_threshold_type="percentile")
```

做法：拆句 → 每句编码成向量 → 算相邻句相似度 → **相似度骤降处下刀**。

```
块数：2，长度 566~1776，平均 1171
覆盖度：2342 / 2343 字符（≈100%，没丢内容）
```

**注意它没有 `chunk_size` 参数**——块长完全由内容决定，所以出现了一个 1776 字符的巨块。

三种取舍：

| | 优点 | 缺点 |
| --- | --- | --- |
| 固定大小 | 快，零依赖 | 大小不可控，短块多 |
| 递归字符 | 大小可控，无模型开销 | 会断句，中文要自己配 separators |
| 语义 | 块内主题单一，检索最干净 | 慢 10 倍以上；块长不可预测；阈值难调 |

**生产上通常不只用语义分块**，而是「递归字符打底 + 结构/语义兜底」。

---

## 五、自己动手验证（建议按顺序做）

### 1. 亲手看到「块大小影响检索」

这是 C2 最重要的实验。改 `03` 的 `CHUNK_SIZE` 跑两次：

```bash
# 先看默认 200
python3 code/C2/03_recursive_character_splitter.py | grep -E "块数"

# 临时改成 100（直接编辑脚本里的 CHUNK_SIZE，或复制一份改）
# → 块数会从 22 涨到 40+，块更碎
```

想更进一步，可以把切好的块接到 C1 的检索里：同一段文本、
`chunk_size=200` 和 `chunk_size=1500` 各建一次索引，问同一个问题，看命中块是否不同。

### 2. 改 `chunk_overlap` 观察重叠

`overlap=10` 和 `overlap=100` 对比：overlap 越大，块数越多、
相邻块重复内容越多（防止答案被切断，但也会引入冗余）。

### 3. 换 `breakpoint_threshold_type`

`04` 里把 `THRESHOLD` 改成 `"standard_deviation"` 或 `"interquartile"`，
看块数怎么变——这就是「阈值难调」的具体体感。

### 4. 换一份文档

`data/C2/txt/` 下换成任意自己的中文 txt，观察：
- 递归分块是否还有碎片块
- 语义分块的块长分布

---

## 六、常见问题

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| `Is poppler installed and in PATH` | 缺 poppler | `brew install poppler` |
| `Created a chunk of size XXX` | 正常警告，不是错误 | 用递归分块可消除 |
| 04 跑得比其他慢很多 | 要加载 bge 模型 + 每句编码 | 正常，实测 ~15 s |
| `File ... does not exist` | 老版本相对路径问题 | 现已修，任意目录都能跑 |
| 想关掉阶段日志 | — | `RAG_LOG=quiet python3 ...` |

---

## 七、C2 和前后章节的关系

```
C1  跑通闭环（分块只是两行代码）
C2  数据准备 ← 你在这一章
     01 解析：文档 → 带类型的元素
     02 固定大小 → 03 递归 → 04 语义（三档递进）
C3  索引构建：这些块怎么变成向量、存进 FAISS/Milvus
C4  检索优化：块切得不好，混合检索/重排也救不回来
```

**一句话总结 C2**：RAG 效果的上限，很大程度上由「块切得好不好」决定；
而块怎么切，取决于你的文档是什么结构、下游检索要什么粒度。
