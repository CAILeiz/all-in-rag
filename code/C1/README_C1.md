# C1 导读：这一章讲什么、要教会你什么、怎么动手验证

> 配套文档：`docs/chapter1/01_RAG_intro.md`（概念）、`02_preparation.md`（环境）、`03_get_start_rag.md`（代码）
> 配套脚本：`code/C1/01_langchain_example.py`、`code/C1/02_llamaIndex_example.py`、`code/C1/lab_walkthrough.py`

---

## 一、C1 讲的是什么

C1 是整本书的**开篇定调**，只干一件事：**用最短的代码把「RAG 是什么」跑给你看**。
它不追求效果、不优化参数，就是要你先建立一个「四步流水线」的肌肉记忆。

三节内容的分工：

| 小节   | 文件                  | 讲什么                                                   |
| ------ | --------------------- | -------------------------------------------------------- |
| 第一节 | `01_RAG_intro.md`     | 概念：RAG 是什么、为什么需要、和微调怎么选、三阶段演进   |
| 第二节 | `02_preparation.md`   | 环境：API Key 怎么申请、Codespaces / Cloud Studio 怎么起 |
| 第三节 | `03_get_start_rag.md` | 代码：用 LangChain 和 LlamaIndex 各写一个 30 行的 RAG    |

### 1. 核心概念（第一节要你记住的）

**RAG = 把「模型脑子里的知识」和「外部资料库的知识」拼起来用。**

- 模型权重里的知识叫**参数化知识**：模糊、会过时、改起来要重训。
- 外部知识库叫**非参数化知识**：精准、能随时更新、改起来只是换个文件。
- RAG 的动作就是：**先检索，再生成**。所以那句比喻很贴切——让 LLM 学会「开卷考试」。

技术上是两个阶段：

```
【离线/建库】文档 → 分块 → 嵌入模型编码成向量 → 存进向量库
【在线/问答】问题 → 同一个嵌入模型编码 → 向量库里找最相似的 top-k 块
           → 把这几块塞进 prompt → LLM 基于它们作答
```

**为什么优先 RAG 而不是微调**，文档给的选型顺序是：

```
提示词工程  →  RAG  →  微调
（改提问）   （补知识）  （改行为/风格）
```

判断标准很简单：想让模型「**知道什么**」用 RAG，想让模型「**怎么做**」才用微调。

### 2. 代码骨架（第三节要你跑通的）

`01_langchain_example.py` 就是这四步的直译，83 行里没有一句废话：

| 步骤       | 代码位置 | 干了什么                                                                       |
| ---------- | -------- | ------------------------------------------------------------------------------ |
| ① 数据准备 | L24–32   | `UnstructuredMarkdownLoader` 读 md，`RecursiveCharacterTextSplitter` 切成 6 块 |
| ② 索引构建 | L35–43   | `bge-small-zh-v1.5` 把每块编码成 512 维向量，存进 `InMemoryVectorStore`        |
| ③ 检索     | L79–80   | `similarity_search(question, k=3)` 取最像的 3 块，拼成一个长字符串             |
| ④ 生成     | L46–82   | 把上下文 + 问题填进 prompt 模板，`llm.invoke()` 拿回答                         |

`02_llamaIndex_example.py` 是同样四步的「低代码版」——索引、检索、拼 prompt 全被 `VectorStoreIndex` 封掉了，
只剩 `index.as_query_engine()` 一句。

### 3. 02 这个脚本到底想教你什么

它不是 01 的「翻译版」，而是用一个更短的脚本回答几个 01 回答不了的问题：

**（1）框架到底替你做了什么**

`VectorStoreIndex.from_documents(documents)` 一行，内部依次做了 3 件事：

```
documents → SentenceSplitter 分片 → embed_model 向量化 → 写进 SimpleVectorStore
```

对照 01 的写法，这对应三行代码和两个步骤：

| LlamaIndex 一行             | LangChain 等价写法                                          |
| --------------------------- | ----------------------------------------------------------- |
| `from_documents()` 里的分片 | `RecursiveCharacterTextSplitter().split_documents(docs)`    |
| 里的向量化                  | `HuggingFaceEmbeddings(...)` + 隐式 `embed_documents()`     |
| 里的写入                    | `InMemoryVectorStore(embeddings)` + `add_documents(chunks)` |

→ **封装不等于魔法**，它只是把你该做的 3 件事挪进了函数内部。

**（2）`Settings` 是全局单例，这是 LlamaIndex 和 LangChain 最大的风格差异**

```python
Settings.llm = OpenAILike(...)          # 全局 LLM
Settings.embed_model = HuggingFaceEmbedding(...)   # 全局嵌入模型
```

设置一次，**之后所有 index / query_engine 都默认用它俩**，不用像 LangChain 那样每个对象都传一遍。
好处是代码干净，代价是**隐式依赖**——脚本一长就不知道某个 index 到底用的哪个模型。
所以调试时可以先打印一下：`print(Settings.llm, Settings.embed_model)`。

**（3）同一个参数名 `chunk_size`，两个框架的单位不一样**

这是跑这两个脚本最容易被坑到的地方：

```
LangChain  : RecursiveCharacterTextSplitter(chunk_size=4000)
             → 4000 个「字符」 → 6 块，每块 ~3804 字符

LlamaIndex : SentenceSplitter(chunk_size=1024)   ← 默认值
             → 1024 个「token」 → 27 块，每块 ~1175 字符（≈926 token）
```

实测这份 22449 字符的 markdown：**1 个 token ≈ 1.27 个中文字符**。
所以 `chunk_size=1024` 实际约等于「1300 字符」，比 LangChain 的 4000 小得多，切出来自然更碎。

> 想验证：`python3 -c "from llama_index.core.node_parser import SentenceSplitter; print(SentenceSplitter()._tokenizer)"`
> 会看到 `cl100k_base` 编码器——和 GPT 系列同一套分词表。

**（4）`as_query_engine()` 默认只取 2 条**

```python
query_engine = index.as_query_engine()                     # 默认 similarity_top_k=2
query_engine = index.as_query_engine(similarity_top_k=3)   # 显式指定
```

脚本里显式写了 `similarity_top_k=3`，否则第 5 步日志显示检索 3 条、第 6 步实际只用 2 条，会对不上。
很多「怎么只回了 2 条」的疑惑都来自这个默认值。

**（5）`Response` 对象带回了检索溯源**

```python
resp = query_engine.query("文中举了哪些例子?")
print(resp)                    # 直接打印 = 答案文本
resp.source_nodes              # 这次用了哪几个 node 作上下文
resp.metadata                  # 每个 node 的来源文件信息
```

这是 LlamaIndex 相对手写流程的一个实际优势：**答案自带引用出处**，
`source_nodes` 里的 `score` 可以拿来判断这次回答可不可信。

---

## 二、这一章要教会你什么

跑通不算学会。C1 真正的教学目标是下面 5 条，每一条都能用脚本验证：

1. **看懂四步流水线**：知道每一步的输入输出是什么，出问题该去查哪一步。
2. **理解「向量」为什么能替代关键词匹配**：问题和原文一个字都不重合，也能检索到。
3. **知道分块策略会影响效果**：`chunk_size` 不是随便填的，切太大主题被稀释、切太小上下文断裂。
4. **知道检索是「永远有返回」的**：即使文档里没答案，`similarity_search` 照样给你 3 块。
   所以「不知道就说不知道」只能靠 prompt 兜底，不能指望相似度阈值。
5. **会用框架，也知道框架帮你省了什么**：LangChain 把每步都摊开给你看，LlamaIndex 把四步压成一句。

---

## 三、怎么跑、怎么验证输出

### 0. 前置检查

```bash
# 大模型端点（本机 Ollama）
curl -s http://127.0.0.1:11434/api/tags | head -c 120

# 依赖是否齐全（能打印版本号就说明装好了）
python3 -c "import langchain, llama_index, sentence_transformers; print('ok')"
```

C1 **不需要 Milvus**（用的是 `InMemoryVectorStore`，跑完就没了），所以不用起 Docker。

### 1. 跑原版脚本（看最终答案）

```bash
# 从任何目录都能跑（脚本会自动把工作目录锚定到 code/C1）
python3 code/C1/01_langchain_example.py
python3 code/C1/02_llamaIndex_example.py

# 或者按教程文档的写法，先进目录
cd code/C1 && python3 01_langchain_example.py
```

> 脚本顶部有一行 `bootstrap.anchor_cwd(__file__)`，会在启动时把工作目录切到脚本所在目录。
> 所以脚本里的 `../../data/C1/xxx` 这类相对路径，不管你在仓库根目录、`code/` 还是 `/tmp`
> 用绝对路径调用，都能正确解析。**这是为了修「换个目录就 File does not exist」而加的。**

输出分两部分，**先看阶段日志，再看最终答案**：

```
[1/6] ▶ 文档开始分片 ...        ← 6 个阶段日志（下一节详解）
...
[6/6] ✓ 完成  3.33 s  | prompt=2050 completion=114 total=2164

========================================================================
  最终回答（answer.content）
========================================================================
文中举了以下例子：
1. 选择餐馆：利用是指我们直接去我们最喜欢的餐馆…
2. 做广告：利用是指我们直接采取最优的广告策略…
3. 挖油：利用是指我们直接在已知的地方挖油…
4. 玩游戏：利用是指我们总是采取某一种策略…
========================================================================
原始返回对象（含 additional_kwargs / response_metadata / id 等）：
content='文中举了以下例子：...' response_metadata={'token_usage': {...}}
```

**怎么判断「跑对了」**：只要回答里出现「餐馆 / 广告 / 挖油 / 游戏」这四个例子中至少 3 个，就说明
加载 → 分块 → 向量化 → 检索 → 生成 这条链是通的。

> `02_llamaIndex_example.py` 现在也会把 `get_prompts()` 那一大坨压成一行摘要
> （`text_qa_template: Context information is below...`），不再刷屏 4000 字符。

### 2. 分阶段日志（脚本自带）

两个主脚本运行时都会打印 6 个阶段，直接照着输出就能看懂流程走到哪了：

```
════════════════════════════════════════════════════════════════════════
  C1 RAG 流水线  ·  共 6 步
  LangChain 版 · 数据准备 → 索引构建 → 检索 → 生成
════════════════════════════════════════════════════════════════════════
[1/6] ▶ 文档开始分片
       func  RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
       file  ../../data/C1/markdown/easy-rl-chapter1.md
       加载完成：1 个 Document，共 22449 字符
[1/6] ✓ 完成  5.15 s  | 6 个 chunk（长度 3657~3934，均 3804）
[2/6] ▶ 开始构建向量
       model      BAAI/bge-small-zh-v1.5
       device     cpu
       normalize  True（L2 归一化，内积即余弦）
[2/6] ✓ 完成  13.75 s  | 向量维度 512，每条 chunk 生成 1 个向量
[3/6] ▶ 开始初始化向量库
       type  InMemoryVectorStore（存在进程内存，脚本结束即消失）
[3/6] ✓ 完成  0 ms  | 实例已创建，此时库内为空
[4/6] ▶ 开始写入向量库
       payload  6 个 chunk → 逐个编码后写入
[4/6] ✓ 完成  260 ms  | 库内条数 = 6（= chunk 数）
[5/6] ▶ llm 相似检索
       question  文中举了哪些例子？
       k         3
       method    InMemoryVectorStore.similarity_search（余弦相似度）
       top1 score=0.3514  第1章 强化学习基础  1.1 强化学习概述…
       top2 score=0.3385  输出：  [ 0.01653398 0.19114579 …]…
       top3 score=0.3341  Q: 基于策略和基于价值的强化学习方法有什么区别?…
[5/6] ✓ 完成  19 ms  | 命中 3 块，拼成上下文 11301 字符
[6/6] ▶ llm 生成回答
       model        deepseek-chat
       endpoint     http://127.0.0.1:11434/v1
       temperature  0.7
       最终 prompt = 11427 字符（上下文 11301 + 模板）
[6/6] ✓ 完成  3.33 s  | prompt=2050 completion=114 total=2164

────────────────────────────────────────────────────────────────────────
  耗时汇总（流水线共 6 步）
    1. 文档开始分片                 5.15 s  █████
    2. 开始构建向量                13.75 s  ███████████████
    3. 开始初始化向量库               0 ms  █
    4. 开始写入向量库               260 ms  █
    5. llm 相似检索                  19 ms  █
    6. llm 生成回答                 3.33 s  ████
────────────────────────────────────────────────────────────────────────
    步骤合计 23.32 s    整轮墙钟 23.32 s
```

**每个阶段都能回答一个「到底发生了什么」的问题：**

| 阶段               | 日志告诉了你什么                                                      | 学到的东西                           |
| ------------------ | --------------------------------------------------------------------- | ------------------------------------ |
| 1 文档开始分片     | 用的是 `RecursiveCharacterTextSplitter`，切出 6 块，每块 3657~3934 字 | 分片是第一步，块大小决定后面检索精度 |
| 2 开始构建向量     | 模型 `bge-small-zh-v1.5`，输出 512 维，已归一化                       | 文本变成 512 个数字                  |
| 3 开始初始化向量库 | `InMemoryVectorStore`，此时**空**                                     | 「建库」和「写入」是两件事           |
| 4 开始写入向量库   | 写入 6 条，库内 = chunk 数                                            | 一条 chunk ↔ 一条向量                |
| 5 llm 相似检索     | 问题、k=3、每块相似度（0.35/0.34/0.33）                               | 检索永不失败，低分也会返回           |
| 6 llm 生成回答     | prompt 11427 字符、模型、token 消耗                                   | 生成质量取决于第 5 步给的上下文      |

**关于「耗时」那两栏有个反直觉的点：**

- 第 2 步「开始构建向量」花了 13.75 秒，第 4 步「写入向量库」只花 260 毫秒。
- 因为第 2 步里包含了**加载模型 + 跑一次探测编码**（首次加载要 10 秒以上），
  真正对 6 个 chunk 做编码是在第 4 步的 `add_documents` 里完成的。
- 所以别被数字骗了：**慢在模型加载，不在向量计算本身**（实测编码 7.6 ms/条）。

`02_llamaIndex_example.py` 的日志会更"反常识"，因为 LlamaIndex 把三步合成了一句：

```
[3/6] ▶ 构建索引
       api   VectorStoreIndex.from_documents(documents)
       note  这一句内部做了 3 件事：分片 → 向量化 → 写入向量库
       分片：共 27 个 node（LlamaIndex 默认 chunk_size=1024）
       向量化：27 条 × 512 维
       写入：SimpleVectorStore（存在内存里）
[3/6] ✓ 完成  1.41 s  | 27 个 node，1 句话干完 LangChain 的 3 步
```

→ 注意 **27 个 node vs LangChain 的 6 个 chunk**：LlamaIndex 默认 `chunk_size=1024`，
所以切得更碎。这就是「框架默认值不一样」的具体体现。

**关掉日志**（只想看答案时）：

```bash
RAG_LOG=quiet python3 01_langchain_example.py
```

### 3. 跑拆解实验（看更细的中间数据）—— 推荐

原版脚本只给你最后一段回答，中间发生了什么完全看不到。`lab_walkthrough.py` 把四步拆开：

```bash
python3 lab_walkthrough.py              # 四步全跑
python3 lab_walkthrough.py 1            # 只看数据准备
python3 lab_walkthrough.py 2            # 只看索引构建
python3 lab_walkthrough.py 3            # 只看检索
python3 lab_walkthrough.py 4            # 只看生成
python3 lab_walkthrough.py 3 --k 5      # 检索改成 top-5
python3 lab_walkthrough.py 3 --chunk-size 500   # 换小分块再检索
```

#### 阶段 1 会告诉你

```
Document 数 : 1
总字符数    : 22449
② 分块完成（chunk_size=4000 / overlap=200）
   块数        : 6
   长度 min/avg/max : 3657 / 3804 / 3934
   chunk[0] len= 3853  第1章 强化学习基础  1.1 强化学习概述…
③ 换个 chunk_size 再切一次（对比用）
   chunk_size=  500 →  71 块，平均   392 字符
   chunk_size= 1500 →  18 块，平均  1332 字符
   chunk_size= 4000 →   6 块，平均  3804 字符
```

→ 记住这个数字对比：**同一份文档，块大小一改，块数从 6 变到 71**。

#### 阶段 2 会告诉你

```
① 嵌入模型就绪：BAAI/bge-small-zh-v1.5
   向量维度    : 512
   向量条数    : 6（每个 chunk 一条）
   第1条前6维  : [-0.0435, 0.0331, 0.0247, -0.0242, -0.0382, -0.0067]
   第1条L2范数 : 1.0000
```

→ 这就是「文本变成数字」的样子。范数 1.0 是因为开了 `normalize_embeddings=True`，
所以后面算内积就等于算余弦相似度。

#### 阶段 3 会告诉你（最值得看）

```
问题: 文中举了哪些例子？
  [1] score=0.3514  len=3853  → 对应 chunk[0]
  [2] score=0.3385  len=3787  → 对应 chunk[5]
  [3] score=0.3341  len=3657  → 对应 chunk[3]
        ★ 这一块里确实出现了例子关键词：['餐馆', '广告', '挖油', '游戏']

拼成上下文：3 块 → 共 11301 字符
对比一下 k 的影响：
   k=1 → 上下文   3853 字符
   k=3 → 上下文  11301 字符
   k=6 → 上下文  22836 字符
```

这里能看出一个**很关键的教学点**：正确答案所在的 `chunk[3]` 只排第 3 名，
第一名是开头的总述——因为 `chunk_size=4000` 时每块太长（约 3800 字符），
一块里混了多个主题，向量被「稀释」，区分度就下来了。

再对比小分块：

```bash
python3 lab_walkthrough.py 3 --chunk-size 500
```

此时 top-1 直接就是讲餐馆的那一段（score 0.4 左右，长度 400 出头），检索精准得多。
**这就是第二章要专门讲分块策略的动机。**

阶段 3 还会做一个反例实验：

```
反例：问一个 markdown 里没有的内容
   『Transformer 的自注意力公式是什么？』 top1 score=0.5003
   → 分数并没有低到能被自动过滤，所以『答不上来』这件事只能靠 prompt 约束。
```

→ 注意这个反例的分数（0.50）**比正例的 top-1（0.35）还高**。
这说明了什么？**余弦相似度的绝对值没有意义，不能用来当「有没有答案」的阈值**。

#### 阶段 4 会告诉你

```
① 最终 prompt 长度：11427 字符
---- prompt 头 300 字符 ----   （能看到模板 + 检索到的原文开头）
---- prompt 尾 120 字符 ----   （能看到「问题: 文中举了哪些例子？ 回答:」）
② 调用 LLM：http://127.0.0.1:11434/v1  model=deepseek-chat
③ 回答（.content）
   文中举了以下例子：1. 选择餐馆… 2. 做广告… 3. 挖油… 4. 玩游戏…
④ 元数据：prompt=2050 completion=98 total=2148
```

→ 这里能看清两件事：
（a）**「检索」到底怎么影响「生成」**——决定最终效果的，是塞进 prompt 的那 11301 个字符；
（b）**为什么本地小模型会重复**：11427 字符的 prompt 被截断/压缩成 2050 token 后，
重复内容没被压掉，于是答案里 1~5 和 6~10 是同一批例子说了两遍。换成云端大模型会好一些。

### 4. 三个课后练习的做法

| 练习                              | 怎么做                                                                                                    | 预期现象                              |
| --------------------------------- | --------------------------------------------------------------------------------------------------------- | ------------------------------------- |
| 只取 `.content`                   | 把 `print(answer)` 改成 `print(answer.content)`                                                           | 输出从一坨元数据变成纯回答            |
| 改 `chunk_size` / `chunk_overlap` | `python3 lab_walkthrough.py 3 --chunk-size 500`，或直接改脚本 L31                                         | 块数、检索命中、回答长度都变          |
| 给 LlamaIndex 加注释              | 逐行读 `02_llamaIndex_example.py`，重点标注 `Settings.llm` / `Settings.embed_model` / `as_query_engine()` | 能说出哪一行等价于 LangChain 的哪一步 |

### 5. 排查常见问题

| 现象                         | 原因                                      | 处理                                                                                   |
| ---------------------------- | ----------------------------------------- | -------------------------------------------------------------------------------------- |
| `nltk` 报 `punkt_tab` 找不到 | 本地 NLTK 数据没配                        | 正常跑脚本会自动加载 `code/.env` 里的 `NLTK_DATA`；若手动跑片段，先 `import bootstrap` |
| `401 invalid key`            | 用了失效的云端 Key                        | 默认已切本地 Ollama，检查 `code/.env` 的 `LLM_BASE_URL`                                |
| `socksio not installed`      | 环境里有 socks 代理                       | `import bootstrap` 会清掉；或 `unset ALL_PROXY`                                        |
| 首次运行特别慢               | 在下载 `bge-small-zh-v1.5`（约 90MB）     | 已缓存在 `.cache/hf`，之后是秒级                                                       |
| 回答跟文档无关               | 检索没命中                                | 跑 `lab_walkthrough.py 3` 看命中块，再调 `k` 或 `chunk_size`                           |
| `File ... does not exist`    | 旧版脚本用相对路径，必须在 `code/C1` 下跑 | 现已修行（`anchor_cwd`），在任意目录都能跑                                             |

---

## 四、C1 在前四章里的位置

```
C1  跑通最小闭环      ← 你现在在这
 ├── 有「加载/分块」的概念，但没细讲   → C2 数据准备（各种格式解析 + 5 种分块策略）
 ├── 有「向量」的概念，但没细讲        → C3 索引构建（嵌入模型、FAISS/Milvus、多模态）
 └── 有「检索」的概念，但只有朴素 top-k  → C4 检索优化（混合检索、自查询、Text2SQL、路由、重排）
```

也就是说：**C1 里的每一步，后面都有一整章来展开**。
所以 C1 阶段不要纠结效果好不好（本地 3B 模型本来就一般），重点是把四步的**因果关系**建立起来：

> 改分块 → 影响向量 → 影响检索命中 → 影响 prompt 内容 → 影响最终回答

这条链能说清楚，C1 就学到位了。
