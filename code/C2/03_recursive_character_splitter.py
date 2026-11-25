"""C2-03 · 递归字符分块（RecursiveCharacterTextSplitter）

对应文档：docs/chapter2/05_text_chunking.md 3.2 节

这一节的要点：**递归分块会「按层级退让」**。
它按 separators 列表顺序尝试：先用 "\\n\\n"（段落）切；某段还是太长，
就退到 "\\n"（行）；再长就退到 "。"（句号）→ "，"（逗号）→ 空格 → 单字符。
所以块长更可控，代价是可能从句子中间断开。

对比 02 的固定大小分块：
    02 → 14 块，长度 10~294，3 块超限
    03 → ?  块，长度 ?~?，（应几乎没有超限）

运行：
    python3 code/C2/03_recursive_character_splitter.py
    RAG_LOG=quiet python3 code/C2/03_recursive_character_splitter.py

3 个阶段：
    [1/3] 加载文档
    [2/3] 递归字符分块
    [3/3] 打印切分结果
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

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

TXT = "../../data/C2/txt/蜂医.txt"
SEPARATORS = ["\n\n", "\n", "。", "，", " ", ""]   # 针对中英文混排的退让顺序
CHUNK_SIZE = 200
CHUNK_OVERLAP = 10

stage = Stage(total=3, title="C2 · 递归字符分块")
stage.header("RecursiveCharacterTextSplitter（按分隔符逐级退让）")

# ---------------------------------------------------------------- 1. 加载
with stage.step("加载文档", loader="TextLoader", file=TXT) as step:
    loader = TextLoader(TXT, encoding="utf-8")
    docs = loader.load()
    step.detail(f"{len(docs)} 个 Document，共 {sum(len(d.page_content) for d in docs)} 字符")
    step.note("原文就绪")

# ---------------------------------------------------------------- 2. 分块
with stage.step(
    "递归字符分块",
    func="RecursiveCharacterTextSplitter",
    params=f"chunk_size={CHUNK_SIZE}, chunk_overlap={CHUNK_OVERLAP}",
    separators=SEPARATORS,
) as step:
    step.detail("切分顺序：先整段 → 不行退到整行 → 再不行退到句号 → 逗号 → 空格 → 单字符")

    text_splitter = RecursiveCharacterTextSplitter(
        # 针对中英文混合文本，定义一个更全面的分隔符列表
        separators=SEPARATORS,   # 按顺序尝试分割
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = text_splitter.split_documents(docs)

    lens = [len(c.page_content) for c in chunks]
    over = [n for n in lens if n > CHUNK_SIZE]
    step.detail(f"块数：{len(chunks)}，长度 {min(lens)}~{max(lens)}，平均 {sum(lens)//len(lens)}")
    if over:
        step.detail(f"⚠ 仍有 {len(over)} 块超过 {CHUNK_SIZE}：{over[:5]}")
        step.detail("  说明连「单字符」都切不开，或 overlap 把相邻片段又拼长了")
    else:
        step.detail(f"✓ 没有超限块（对比 02 的 3 块超限）")
    step.note(f"{len(chunks)} 块，{len(over)} 块超限")

# ---------------------------------------------------------------- 3. 打印
with stage.step("打印切分结果", note="原文输出，便于肉眼检查边界") as step:
    step.detail("以下为脚本原始输出（前 5 块）")

    print(f"文本被切分为 {len(chunks)} 个块。\n")
    print("--- 前5个块内容示例 ---")
    for i, chunk in enumerate(chunks[:5]):
        print("=" * 60)
        print(f'块 {i+1} (长度: {len(chunk.page_content)}): "{chunk.page_content}"')

    step.note("完成")

stage.summary()

print()
print("=" * 72)
print("  观察点：和 02 对比")
print("=" * 72)
print(f"  02 固定大小 : 14 块，长度 10~294，3 块超限")
print(f"  03 递归     : {len(chunks)} 块，长度 {min(lens)}~{max(lens)}，{len(over)} 块超限")
print()
print("  递归分块切得更多但更均匀，因为它会不断退让直到塞得下。")
print("  代价：可能从句子中间断开（看块 3 那种以「。」开头的碎片）。")
print()
print("→ 语义分块（04）不再按长度切，而是按「话题是否变了」切")
print("=" * 72)
