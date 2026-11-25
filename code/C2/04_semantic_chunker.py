"""C2-04 · 语义分块（SemanticChunker）

对应文档：docs/chapter2/05_text_chunking.md 3.3 节

这一节的要点：**前两种都按「长度」切，语义分块按「意思变没变」切。**

做法：把文本拆成句子 → 用嵌入模型给每句算向量 → 算相邻句子的相似度 →
相似度突然掉下去的地方（超过阈值）就是话题边界，在那里下刀。
所以块长完全由内容决定，可能 200 字，也可能 1500 字。

参数 breakpoint_threshold_type 决定「掉多少算突变」：
    percentile         （默认）以所有相邻相似度的某个百分位为界
    standard_deviation 以均值 + N 倍标准差为界
    interquartile      以四分位距为界
    gradient           以相似度变化的梯度为界

运行（要加载嵌入模型，约 15 秒）：
    python3 code/C2/04_semantic_chunker.py
    RAG_LOG=quiet python3 code/C2/04_semantic_chunker.py

4 个阶段：
    [1/4] 加载文档
    [2/4] 初始化嵌入模型（语义分块靠它判断话题）
    [3/4] 语义分块
    [4/4] 打印切分结果
"""
import os as _os, sys as _sys
# 向上找到含 bootstrap.py 的 code/ 目录，加入 sys.path（脚本可能在多级子目录里）
_p = _os.path.dirname(_os.path.abspath(__file__))
while _p != _os.path.dirname(_p) and not _os.path.exists(_os.path.join(_p, "bootstrap.py")):
    _p = _os.path.dirname(_p)
_sys.path.insert(0, _p)
import bootstrap  # noqa: F401  仅为了副作用：加载 code/.env、固定缓存目录、修代理与证书
bootstrap.anchor_cwd(__file__)   # 工作目录钉在脚本所在目录，相对路径才不会失效
from stage_log import Stage

from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader

TXT = "../../data/C2/txt/蜂医.txt"
THRESHOLD = "percentile"   # 也可换成 "standard_deviation" / "interquartile" / "gradient"

stage = Stage(total=4, title="C2 · 语义分块")
stage.header("SemanticChunker（按相邻句子相似度的突变处切）")

# ---------------------------------------------------------------- 1. 加载
with stage.step("加载文档", loader="TextLoader", file=TXT) as step:
    loader = TextLoader(TXT, encoding="utf-8")
    documents = loader.load()
    step.detail(f"{len(documents)} 个 Document，共 {sum(len(d.page_content) for d in documents)} 字符")
    step.note("原文就绪")

# ---------------------------------------------------------------- 2. 嵌入模型
with stage.step(
    "初始化嵌入模型",
    model="BAAI/bge-small-zh-v1.5",
    device="cpu",
    why="语义分块要算「相邻句子的相似度」，所以必须先有嵌入模型",
) as step:
    step.detail("⚠ 这是 02/03 不需要的额外开销：多了模型加载 + 每句一次编码")
    step.detail("  RAG_LOG 里第 2 步耗时明显更长，就是因为这个")

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    step.note("模型就绪")

# ---------------------------------------------------------------- 3. 分块
with stage.step(
    "语义分块",
    func="SemanticChunker",
    threshold=THRESHOLD,
    note="没有 chunk_size —— 块长完全由内容决定",
) as step:
    # 初始化 SemanticChunker
    text_splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type=THRESHOLD
    )

    docs = text_splitter.split_documents(documents)

    lens = [len(c.page_content) for c in docs]
    raw_len = sum(len(d.page_content) for d in documents)
    step.detail(f"块数：{len(docs)}，长度 {min(lens)}~{max(lens)}，平均 {sum(lens)//len(lens)}")
    step.detail(f"覆盖度：{sum(lens)} / {raw_len} 字符（应≈100%，说明没丢内容）")
    step.note(f"{len(docs)} 块")

# ---------------------------------------------------------------- 4. 打印
with stage.step("打印切分结果", note="原文输出（前 2 块）") as step:
    step.detail("以下为脚本原始输出")

    print(f"文本被切分为 {len(docs)} 个块。\n")
    print("--- 前2个块内容示例 ---")
    for i, chunk in enumerate(docs[:2]):
        print("=" * 60)
        print(f'块 {i+1} (长度: {len(chunk.page_content)}):\n"{chunk.page_content}"')

    step.note("完成")

stage.summary()

print()
print("=" * 72)
print("  三种策略对比（同一份 2343 字符的文本）")
print("=" * 72)
print("  02 固定大小  : 14 块，长度 10~294，3 块超限  ← 大小不可控")
print("  03 递归字符  : 22 块，长度 3~200，0 块超限   ← 大小可控但会断句")
print(f"  04 语义      : {len(docs)} 块，长度 {min(lens)}~{max(lens)}"
      f"              ← 块长由话题决定")
print()
print("  语义分块的取舍：")
print("    好：块内主题单一，检索命中时上下文最干净")
print("    坏：① 多一次模型开销（慢 10 倍以上）")
print("        ② 块长不可预测，可能超出下游模型的上下文预算")
print("        ③ 阈值难调，不同文本要重新试")
print()
print("  生产上常见做法：递归字符分块打底 + 语义/结构做兜底，而不是纯语义分块。")
print("=" * 72)
