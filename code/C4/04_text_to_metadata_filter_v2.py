"""C4-04 · 自然语言 → 元数据过滤 v2：LLM 出 JSON 指令，代码里排序

对应文档：docs/chapter4/02_text_to_metadata_filter.md（与 02 对照阅读）

和 02 的差别：

    02    SelfQueryRetriever：LLM 出结构化 filter，交给向量库执行
    04v2  自己写 prompt：LLM 只出 {"sort_by": ..., "order": ...}，排序在 Python 里做

v2 的好处：完全可控，不受向量库支持哪些算子限制；
代价：得自己写 prompt、自己校验返回的 JSON，字段一多 prompt 就会膨胀。

运行（需要能访问 B 站接口）：
    python3 code/C4/04_text_to_metadata_filter_v2.py
    RAG_LOG=quiet python3 code/C4/04_text_to_metadata_filter_v2.py

5 个阶段：
    [1/5] 加载 B 站视频元数据
    [2/5] 建 Chroma 向量库
    [3/5] 声明可过滤字段
    [4/5] 取出全部文档（用于内存排序）
    [5/5] 用 LLM 生成排序指令并执行
"""
import os as _os, sys as _sys

# 向上找到含 bootstrap.py 的 code/ 目录，加入 sys.path（脚本可能在多级子目录里）
_p = _os.path.dirname(_os.path.abspath(__file__))
while _p != _os.path.dirname(_p) and not _os.path.exists(_os.path.join(_p, "bootstrap.py")):
    _p = _os.path.dirname(_p)
_sys.path.insert(0, _p)
import bootstrap  # noqa: F401  仅为了副作用：加载 code/.env、修代理与证书

bootstrap.anchor_cwd(__file__)  # 工作目录钉在脚本所在目录，相对路径才不会失效
from llm_config import openai_client_kwargs
from stage_log import Stage

import json
import logging

from langchain.chains.query_constructor.base import AttributeInfo
from langchain_community.document_loaders import BiliBiliLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI

logging.basicConfig(level=logging.INFO)

VIDEO_URLS = [
    "https://www.bilibili.com/video/BV1Bo4y1A7FU",
    "https://www.bilibili.com/video/BV1ug4y157xA",
    "https://www.bilibili.com/video/BV1yh411V7ge",
]

QUERIES = ["时间最短的视频", "播放量最高的视频"]

SORT_PROMPT = """你是一个智能助手，请将用户的问题转换成一个用于排序视频的JSON指令。

你需要识别用户想要排序的字段和排序方向。
- 排序字段必须是 'view_count' (观看次数) 或 'length' (时长) 之一。
- 排序方向必须是 'asc' (升序) 或 'desc' (降序) 之一。

例如:
- '时间最短的视频' 或 '哪个视频时间最短' 应转换为 {{"sort_by": "length", "order": "asc"}}
- '播放量最高的视频' 或 '哪个视频最火' 应转换为 {{"sort_by": "view_count", "order": "desc"}}

请根据以下问题生成JSON指令:
原始问题: "{query}"

JSON指令:"""

stage = Stage(total=5, title="C4 · 元数据过滤 v2（LLM 出 JSON 排序指令）")
stage.header("LLM 只负责翻译意图，排序交给 Python —— 边界清晰、可控")

# ---------------------------------------------------------------- 1. 加载视频
bili = []
with stage.step(
    "加载 B 站视频元数据",
    count=len(VIDEO_URLS),
    why="title/author/view_count/length 是后面能拿来排序的字段",
) as step:
    try:
        loader = BiliBiliLoader(video_urls=VIDEO_URLS)
        docs = loader.load()
        for doc in docs:
            original = doc.metadata
            doc.metadata = {
                "title": original.get("title", "未知标题"),
                "author": original.get("owner", {}).get("name", "未知作者"),
                "source": original.get("bvid", "未知ID"),
                "view_count": original.get("stat", {}).get("view", 0),
                "length": original.get("duration", 0),
            }
            bili.append(doc)
    except Exception as e:
        step.detail(f"加载 BiliBili 视频失败：{e}")

    if not bili:
        step.detail("没有成功加载任何视频，程序退出")
        step.note("已中止")
        stage.summary()
        raise SystemExit(1)

    for d in bili:
        step.detail(
            f"  · {d.metadata['title'][:28]}  "
            f"{d.metadata['length']}s  {d.metadata['view_count']} 次"
        )
    step.note(f"{len(bili)} 个视频")

# ---------------------------------------------------------------- 2. 向量库
with stage.step(
    "建 Chroma 向量库",
    embed="BAAI/bge-small-zh-v1.5",
    why="这个 demo 的排序其实不依赖向量，但保留向量库是为了后面接语义召回",
) as step:
    embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
    vectorstore = Chroma.from_documents(bili, embed_model)
    step.note(f"{len(bili)} 条")

# ---------------------------------------------------------------- 3. 字段声明
with stage.step(
    "声明可过滤字段",
    fields="title / author / view_count / length",
    why="prompt 里只允许 sort_by 取 view_count 或 length，字段类型必须和元数据一致",
) as step:
    metadata_field_info = [
        AttributeInfo(name="title", description="视频标题（字符串）", type="string"),
        AttributeInfo(name="author", description="视频作者（字符串）", type="string"),
        AttributeInfo(name="view_count", description="视频观看次数（整数）", type="integer"),
        AttributeInfo(name="length", description="视频长度（整数）", type="integer"),
    ]
    step.detail("view_count / length 是 integer，排序才有意义")
    step.note("4 个字段")

# ---------------------------------------------------------------- 4. 取全部文档
with stage.step(
    "取出全部文档",
    k=len(bili),
    why="要排序就得先把候选集全拿出来，排序发生在内存里而不是向量库里",
) as step:
    all_documents = vectorstore.similarity_search("", k=len(bili))
    step.detail(f"取回 {len(all_documents)} 条")
    step.note(f"{len(all_documents)} 条")

# ---------------------------------------------------------------- 5. 查询
with stage.step(
    "LLM 生成排序指令并执行",
    count=len(QUERIES),
    why="注意 response_format=json_object：强制 LLM 返回可解析的 JSON，比让它自由发挥稳",
) as step:
    client = OpenAI(**openai_client_kwargs())

    for query in QUERIES:
        step.detail("")
        step.detail(f"--- 原始查询: '{query}' ---")

        prompt = SORT_PROMPT.format(query=query)
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )

        try:
            instruction = json.loads(response.choices[0].message.content)
            step.detail(f"  生成指令：{instruction}")

            sort_by = instruction.get("sort_by")
            order = instruction.get("order")

            if sort_by in ["length", "view_count"] and order in ["asc", "desc"]:
                reverse_order = order == "desc"
                sorted_docs = sorted(
                    all_documents,
                    key=lambda doc: doc.metadata.get(sort_by, 0),
                    reverse=reverse_order,
                )
                if sorted_docs:
                    doc = sorted_docs[0]
                    step.detail(
                        f"  Top1：{doc.metadata.get('title', '未知')[:28]}  "
                        f"{doc.metadata.get('length', '?')}s  "
                        f"{doc.metadata.get('view_count', '?')} 次"
                    )
                else:
                    step.detail("  没有找到任何视频")
            else:
                step.detail(f"  生成的指令无效（sort_by={sort_by}, order={order}），跳过")

        except (json.JSONDecodeError, KeyError) as e:
            step.detail(f"  解析或执行指令失败：{e}")

    step.note(f"{len(QUERIES)} 个查询")

stage.summary()

print()
print("=" * 72)
print("  观察点：和 02 对比")
print("=" * 72)
print("1. 02 把过滤交给向量库；04v2 把排序拿回 Python —— 后者能做任意复杂逻辑")
print("2. response_format=json_object 是关键：不强制的话 LLM 常加解释文字，解析就崩")
print("3. 一定要校验 sort_by/order 白名单，否则 LLM 返回非法字段会让排序静默出错")
print("4. 这个方案不依赖向量库能力，但 prompt 要自己维护，字段多了容易失控")
print()
print("提示：sorted() 的 key 用了 .get(sort_by, 0)，缺字段时按 0 处理而不是抛异常")
print("=" * 72)
