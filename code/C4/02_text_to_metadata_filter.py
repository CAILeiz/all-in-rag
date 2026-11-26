"""C4-02 · 自然语言 → 元数据过滤（SelfQueryRetriever）

对应文档：docs/chapter4/02_text_to_metadata_filter.md

要点：向量检索只会「找相似」，但「时间最短的视频」这种需求本质是**结构化查询**。
SelfQueryRetriever 的做法是先用一次 LLM 调用把自然语言拆成两部分：

    语义 query  →  真正拿去算向量相似度的文本
    metadata filter  →  交给向量库的 expr / where 条件

运行（需要能访问 B 站接口，用来拉视频元数据）：
    python3 code/C4/02_text_to_metadata_filter.py
    RAG_LOG=quiet python3 code/C4/02_text_to_metadata_filter.py

5 个阶段：
    [1/5] 加载 B 站视频元数据
    [2/5] 建 Chroma 向量库
    [3/5] 声明可过滤字段
    [4/5] 创建 SelfQueryRetriever
    [5/5] 执行查询
"""
import os as _os, sys as _sys
# 向上找到含 bootstrap.py 的 code/ 目录，加入 sys.path（脚本可能在多级子目录里）
_p = _os.path.dirname(_os.path.abspath(__file__))
while _p != _os.path.dirname(_p) and not _os.path.exists(_os.path.join(_p, "bootstrap.py")):
    _p = _os.path.dirname(_p)
_sys.path.insert(0, _p)
import bootstrap  # noqa: F401  仅为了副作用：加载 code/.env、修代理与证书
bootstrap.anchor_cwd(__file__)   # 工作目录钉在脚本所在目录，相对路径才不会失效
from llm_config import llm_kwargs
from stage_log import Stage

import os
from langchain_deepseek import ChatDeepSeek 
from langchain_community.document_loaders import BiliBiliLoader
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import logging

logging.basicConfig(level=logging.INFO)

stage = Stage(total=5, title="C4 · 自然语言 → 元数据过滤（自查询检索）")
stage.header("LLM 把「时长大于600秒」翻译成结构化 filter，再去向量库查询")

# 1. 初始化视频数据
video_urls = [
    "https://www.bilibili.com/video/BV1Bo4y1A7FU", 
    "https://www.bilibili.com/video/BV1ug4y157xA",
    "https://www.bilibili.com/video/BV1yh411V7ge",
]

bili = []
with stage.step(
    "加载 B 站视频元数据",
    count=len(video_urls),
    why="这些视频的 title/author/view_count/length 就是后面能被「过滤」的字段",
) as step:
  try:
    loader = BiliBiliLoader(video_urls=video_urls)
    docs = loader.load()
    
    for doc in docs:
        original = doc.metadata
        
        # 提取基本元数据字段
        metadata = {
            'title': original.get('title', '未知标题'),
            'author': original.get('owner', {}).get('name', '未知作者'),
            'source': original.get('bvid', '未知ID'),
            'view_count': original.get('stat', {}).get('view', 0),
            'length': original.get('duration', 0),
        }
        
        doc.metadata = metadata
        bili.append(doc)
  except Exception as e:
    step.detail(f"加载 BiliBili 视频失败：{e}")
  if not bili:
    step.detail("没有成功加载任何视频，程序退出")
    raise SystemExit(1)
  for d in bili:
    step.detail(f"  · {d.metadata['title'][:30]}  {d.metadata['length']}s  {d.metadata['view_count']} 次")
  step.note(f"{len(bili)} 个视频")

# 2. 创建向量存储
with stage.step(
    "建 Chroma 向量库",
    embed="BAAI/bge-small-zh-v1.5",
    why="元数据过滤仍建在向量库上：先按 filter 筛，再算相似度",
) as step:
    embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
    vectorstore = Chroma.from_documents(bili, embed_model)
    step.note(f"{len(bili)} 条")

# 3. 配置元数据字段信息
with stage.step(
    "声明可过滤字段",
    fields="title / author / view_count / length",
    why="LLM 只能按这里声明过的字段生成过滤条件，没声明的字段它不会用",
) as step:
    metadata_field_info = [
        AttributeInfo(
            name="title",
            description="视频标题（字符串）",
            type="string", 
        ),
        AttributeInfo(
            name="author",
            description="视频作者（字符串）",
            type="string",
        ),
        AttributeInfo(
            name="view_count",
            description="视频观看次数（整数）",
            type="integer",
        ),
        AttributeInfo(
            name="length",
            description="视频长度（整数）",
            type="integer"
        )
    ]
    step.detail("view_count / length 是 integer —— 类型声明错了，比较运算就会失败")
    step.note("4 个字段")

# 4. 创建自查询检索器
with stage.step(
    "创建 SelfQueryRetriever",
    api="SelfQueryRetriever.from_llm(...)",
    why="它内部会先调一次 LLM，把自然语言拆成「语义查询 + 结构化 filter」",
) as step:
    llm = ChatDeepSeek(temperature=0, **llm_kwargs())
    retriever = SelfQueryRetriever.from_llm(
        llm=llm,
        vectorstore=vectorstore,
        document_contents="记录视频标题、作者、观看次数等信息的视频元数据",
        metadata_field_info=metadata_field_info,
        enable_limit=True,
        verbose=True,
    )
    step.note("已创建")

# 5. 执行查询示例
queries = [
    "时间最短的视频",
    "时长大于600秒的视频"
]

with stage.step(
    "执行查询",
    count=len(queries),
    why="verbose=True 会打印它生成的结构化查询，注意看 filter 长什么样",
) as step:
    for query in queries:
        step.detail("")
        step.detail(f"--- 查询: '{query}' ---")
        results = retriever.invoke(query)
        if not results:
            step.detail("未找到匹配的视频")
            continue
        for doc in results:
            step.detail(
                f"  {doc.metadata.get('title', '未知')[:28]}  "
                f"{doc.metadata.get('length', '?')}s  "
                f"{doc.metadata.get('view_count', '?')} 次"
            )
    step.note(f"{len(queries)} 个查询")

stage.summary()

print()
print("=" * 72)
print("  观察点")
print("=" * 72)
print("1. 「时间最短的视频」→ LLM 生成排序/过滤条件，不是靠向量相似度猜")
print("2. 元数据字段的 type 必须写对：length 若声明成 string，> 600 就无法比较")
print("3. 这类能力让向量库支持「结构化条件 + 语义」混合查询")
print("4. 代价：每次查询多一次 LLM 调用，且字段多时 prompt 会变长")
print()
print("提示：依赖 B 站接口，网络不通时会提示「没有成功加载任何视频」")
print("=" * 72)
