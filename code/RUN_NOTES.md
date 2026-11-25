# C1–C4 本地跑通记录

本文件记录「不开云 API Key、全部走本机 Ollama + Docker Milvus」跑通 `code/C1` ~ `code/C4` 的过程、
做过的改动、以及一键复现命令。

## 1. 一键复现

```bash
# 1) 启动 Milvus（C3-04 / C4-01 / C4-03 需要 localhost:19530）
docker compose -f code/docker-compose.local.yml up -d

# 2) 顺序跑 C1~C4（也可只跑某几章：bash code/run_all.sh C2 C3）
bash code/run_all.sh
```

日志在 `code/logs/<章节>/<脚本>.log`，汇总在 `code/logs/summary.txt`。
最近一次结果：**25/25 PASS，0 FAIL**。

单跑某一个脚本（两种都可以）：

```bash
# A. 最省事：脚本自带 bootstrap，直接跑
cd code/C2 && python3 04_semantic_chunker.py

# B. 想在同一 shell 里连续跑多个脚本
source code/local_env.sh
cd code/C2 && python3 04_semantic_chunker.py
```

## 2. 本机环境事实

| 项 | 值 |
| --- | --- |
| Python | 3.10.4（`/Library/Frameworks/Python.framework/Versions/3.10`） |
| 机器 | Apple M1 Pro / 16 GB |
| 大模型 | Ollama `http://127.0.0.1:11434/v1`，别名 `deepseek-chat` → `qwen2.5-coder:3b` |
| 向量库 | Docker Milvus v2.5.14 standalone（19530 / 9091） |
| 模型缓存 | 仓库内 `.cache/hf`（bge-small-zh、bge-base-en、bge-large-zh、bge-m3、siglip、bert-base-uncased） |
| NLTK 数据 | 仓库内 `.cache/nltk` |

> 说明：原先 `DEEPSEEK_API_KEY`（`sk-FfLl…RHe4`）在 DeepSeek 和 AIHubmix 都返回 **401 invalid key**，
> 所以按「用本地已启动的模型」方案，全部改成走 Ollama 的 OpenAI 兼容端点，不依赖任何云端 Key。

## 3. 新增文件

| 文件 | 作用 |
| --- | --- |
| `code/local_env.sh` | 交互式一键环境：LLM 指向本地 Ollama、HF/NLTK/llama_index 缓存落在仓库 `.cache/`、修好代理与 CA 证书 |
| `code/bootstrap.py` | 脚本级引导：直接 `python3 xxx.py` 时也会加载 `code/.env`、修代理与证书（放在各脚本顶部） |
| `code/llm_config.py` | 统一读取 `LLM_BASE_URL / LLM_MODEL / LLM_API_KEY`，供各脚本复用 |
| `code/compat_shims.py` | 绕过 `mistralai 3.x` 命名空间包导致 `llama_index.experimental` 导入失败的问题 |
| `code/run_all.sh` | 顺序执行 C1~C4 的 25 个脚本并写日志、汇总 PASS/FAIL |
| `code/docker-compose.local.yml` | 本机可拉取的 Milvus 编排（替代官方 compose 里拉不动的镜像） |
| `code/.env` | 可选：把 `LLM_*` / `DEEPSEEK_*` 指向本地端点（已 gitignore） |
| `code/curl_deepseek.sh` | curl 直调 `deepseek-chat`（本地/云端可切），支持 `--raw` |
| `code/inspect_vectors.py` | 巡检 Milvus / FAISS / LlamaIndex / Chroma 里实际存的向量与原文 |
| `code/demo_write_vectors.py` | 往 Milvus 写 4 个可留存的 `demo_*` collection（教程脚本会 drop，所以另备一份） |
| `code/demo_embedding.py` | 逐个演示 5 类 embedding：维度、范数、相似度 |
| `code/VECTORS_GUIDE.md` | 上述三个工具的用法与「用了哪些 embedding」的完整说明 |

## 4. 修改的脚本（都只是把硬编码的云端地址改成可配置）

统一模式：脚本**最顶部**插入引导块（必须在 transformers / langchain / openai 等重依赖之前，
否则 `HF_HOME`、`SSL_CERT_FILE` 这类变量会太晚生效）：

```python
import os as _os, sys as _sys
_p = _os.path.dirname(_os.path.abspath(__file__))
while _p != _os.path.dirname(_p) and not _os.path.exists(_os.path.join(_p, "bootstrap.py")):
    _p = _os.path.dirname(_p)
_sys.path.insert(0, _p)
import bootstrap  # noqa: F401  仅为了副作用：加载 code/.env、固定缓存目录、修代理与证书
from llm_config import llm_kwargs, openai_llm_kwargs, openai_client_kwargs, LLM_MODEL
```

不带 `llm_config` 的纯检索/嵌入脚本（C2-04、C3-01~04、C3/work_*、C4-01*、C4-03_v2、C4/work_rerank）
只插入前 8 行 `import bootstrap` 部分，用来固定缓存目录。

然后把写死的 `model="glm-4.7-flash-free", base_url="https://aihubmix.com/v1", api_key=os.getenv(...)`
换成 `**llm_kwargs()` / `**openai_llm_kwargs()` / `**openai_client_kwargs()`。

涉及脚本：

- 换成 `llm_config`（会调大模型）：`C1/01`、`C1/02`、`C3/05`、`C3/06`、`C3/07`、`C4/02`、`C4/04`、
  `C4/05`、`C4/06`、`C4/07`、`C4/text2sql/sql_generator.py`
- 只加 `import bootstrap`（固定缓存/证书，逻辑零改动）：`C2/04`、`C3/01`、`C3/02`、`C3/03`、`C3/04`、
  `C3/work_multimodal_dragon_search.py`、`C3/work_hybrid_multimodal_search.py`、`C4/01`、`C4/01_v2`、
  `C4/03_text2sql_demo_v2.py`、`C4/work_rerank_and_refine.py`

**不改默认值也能跑云端**：删掉 `code/.env`（或清空 `LLM_*`）后，`llm_config` 回落到
`https://aihubmix.com/v1` + `glm-4.7-flash-free`，脚本行为与教程原版一致。

## 5. 为跑通做的其它修正

1. **Milvus 镜像**：`quay.io/coreos/etcd:v3.5.16` 与 `minio/minio:RELEASE.2023-03-20…` 当前网络下拉不到
   （quay 连接被重置、minio 该 tag 已下架），本地编排换成
   `rancher/mirrored-coreos-etcd:v3.5.16` + `bitnamilegacy/minio:latest`，
   并把 minio 宿主端口挪到 `19000/19001`（9000/9001 已被既有容器占用）。
2. **Text2SQL 知识库补全**（`code/C4/text2sql/data/`）：演示库 `users` 表有 `city` 列，但知识库里没有，
   导致「统计每个城市的用户数量」一直被生成成 `FROM cities` 而报 `no such table: cities`。
   已在 `qsql_examples.json` 补一条示例，并在 `ddl_examples.json` / `db_descriptions.json` 的 `users` 补 `city` 列。
3. **NLTK**：`unstructured` 与 `llama_index` 会尝试联网下载 nltk 数据（本机走代理会触发 SSRF 保护报
   `Security Violation`）。已把 punkt/punkt_tab/stopwords/tagger/wordnet/omw 预置到 `.cache/nltk`，
   并给 `llama_index` 期待的 `_static/nltk_cache` 做软链。
4. **代理与证书**：环境里的 `ALL_PROXY=socks5://…` 会让 httpx/openai 报 `socksio not installed`，
   `NO_PROXY` 里的 `[::1]` 写法会抛 `InvalidURL: Invalid port: ':1]'`；`local_env.sh` 里统一清掉 socks
   代理、重写 no_proxy，并把 `SSL_CERT_FILE/REQUESTS_CA_BUNDLE` 指向 certifi，修掉 bilibili 请求的
   `CERTIFICATE_VERIFY_FAILED`。
5. **poppler**：`unstructured` 解析 PDF 需要 `pdftoppm`，已 `brew install poppler`。
6. **mistralai 兼容**：见 `compat_shims.py`。
8. **C4/07 的 LLM 压缩返回空结果**：`LLMChainExtractor` 自带的提示词是英文
   （`"...extract any part of the context AS IS..."`），本地 3B 模型遇到它稳定输出空串，
   于是「压缩后结果」一条不剩，但脚本 exit 0 掩盖了问题。
   已换成中文提示词，并加 `ChineseExtractCompressor` 过滤「无」这类占位输出。
   英文提示词 0/5 命中 → 中文提示词 2/5 命中。

7. **相对路径失效**：所有脚本里写的是 `../../data/Cx/...`，只有 `cd code/Cx` 后才对，
   从仓库根目录直接 `python3 code/C1/02_llamaIndex_example.py` 会报
   `ValueError: File ../../data/C1/... does not exist`。
   已在 `code/bootstrap.py` 加 `anchor_cwd(__file__)`（启动时把 CWD 钉到脚本目录），
   并在 27 个脚本里调用，现在**任意目录都能跑**，且产物仍落在原来的位置。

## 6. 各章运行结果摘要

| 脚本 | 结果 |
| --- | --- |
| `C1/01_langchain_example.py` | 正确答案：餐馆、做广告、挖油、玩游戏四个例子 |
| `C1/02_llamaIndex_example.py` | 同上（LlamaIndex 版） |
| `C2/01_unstructured_example.py` | 279 个元素 / 7500 字符，Header/Title/NarrativeText/ListItem 等分类正常 |
| `C2/02_character_splitter.py` | 14 块（固定长度切分） |
| `C2/03_recursive_character_splitter.py` | 22 块（递归分隔符切分） |
| `C2/04_semantic_chunker.py` | 2 块（语义切分） |
| `C3/01_bge_visualized.py` | 多模态相似度：图文1 vs 图文2 = 0.9058 最高 |
| `C3/02_langchain_faiss.py` | FAISS 落盘/加载/检索正常，命中「FAISS 是做什么的」 |
| `C3/03_llamaindex_vector.py` | 索引持久化到 `./llamaindex_index_store` |
| `C3/04_multi_milvus.py` | 7 张图入库，Top1 = `query.png` 自身(0.9466)，Top2 = `dragon02.png`(0.7443)，可视化已保存 |
| `C3/05_sentence_window_retrieval.py` | AMOC 问题，句子窗口 vs 常规检索两路回答都正确 |
| `C3/06_recursive_retrieval.py` | 路由到 `年份_1994`，PandasQueryEngine 正确执行 |
| `C3/07_recursive_retrieval_v2.py` | 路由 + 元数据过滤两步式检索正常 |
| `C3/work_multimodal_dragon_search.py` | 图+文/纯文/纯图三种查询均返回合理结果 |
| `C3/work_hybrid_multimodal_search.py` | multimodal/dense/sparse/hybrid 四路对比 + 4 张可视化图 |
| `C4/01_hybrid_search.py` | BGE-M3 稠密/稀疏/RRF 混合检索，「悬崖上的白龙」稳定 Top1 |
| `C4/01_hybrid_search_v2.py` | SigLIP + TF-IDF 版本同样跑通 |
| `C4/02_text_to_metadata_filter.py` | SelfQueryRetriever 正确解析「时间最短」「时长>600 秒」 |
| `C4/03_text2sql_demo.py` | 6 个问题全部生成正确 SQL（修复 `cities` 后 6/6） |
| `C4/03_text2sql_demo_v2.py` | 纯 SQL 演示正常 |
| `C4/04_text_to_metadata_filter_v2.py` | LLM 生成 `{"sort_by":"length","order":"asc"}` 等指令并排序 |
| `C4/05_llm_based_routing.py` | 川菜/粤菜/其他三路路由 + 回答 |
| `C4/06_embedding_based_routing.py` | 嵌入相似度路由，水煮鱼→川菜、云吞面→粤菜 |
| `C4/07_rerank_and_refine.py` | ColBERT 重排 + LLM 压缩，最终输出精炼句 |
| `C4/work_rerank_and_refine.py` | 同上变体 |

## 7. 已知限制

- 本地模型是 `qwen2.5-coder:3b`（3B 级别），比教程用的云端大模型弱：
  个别开放式问答的措辞/路由判断会有偏差（例如 `C4/05` 把「番茄炒蛋」判成粤菜）。
  代码链路本身是通的，换回云端 Key 后结果质量会提升。
- `qwen2.5-coder:3b` 在 `C3/06` 的 Pandas 指令生成里给出的是「最小评分人数」，
  与自然语言问法一致，但最终复述时把片名说成了「燃烧岁月」（实际 1994 年该片为《燃情岁月》），属于小模型幻觉。
- Milvus 相关脚本结束时会 `drop_collection`，不会残留数据；`code/volumes/` 只有 Milvus 自身元数据。
