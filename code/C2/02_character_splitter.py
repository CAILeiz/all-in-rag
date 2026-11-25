"""C2-02 · 固定大小分块（CharacterTextSplitter）

对应文档：docs/chapter2/05_text_chunking.md 3.1 节

这一节的要点：**「固定大小」其实保证不了大小**。
CharacterTextSplitter 只在分隔符（默认 "\n\n"）处切，遇到一整段超过 chunk_size 的文字时，
它既不能把段落劈开，也不肯丢弃，只能原样保留 —— 于是你会看到
"Created a chunk of size 277, which is longer than the specified 200" 这种警告。

运行：
    python3 code/C2/02_character_splitter.py
    RAG_LOG=quiet python3 code/C2/02_character_splitter.py   # 只要原始输出

3 个阶段：
    [1/3] 加载文档
    [2/3] 固定大小分块
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

from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader

TXT = "../../data/C2/txt/蜂医.txt"
CHUNK_SIZE = 200
CHUNK_OVERLAP = 10

stage = Stage(total=3, title="C2 · 固定大小分块")
stage.header("CharacterTextSplitter（按分隔符切，不保证大小上限）")

# ---------------------------------------------------------------- 1. 加载
with stage.step("加载文档", loader="TextLoader", file=TXT) as step:
    # 1. 文档加载
    loader = TextLoader(TXT, encoding="utf-8")
    docs = loader.load()
    raw_len = sum(len(d.page_content) for d in docs)
    step.detail(f"{len(docs)} 个 Document，共 {raw_len} 字符")
    step.note("原文就绪")

# ---------------------------------------------------------------- 2. 分块
with stage.step(
    "固定大小分块",
    func="CharacterTextSplitter",
    params=f"chunk_size={CHUNK_SIZE}, chunk_overlap={CHUNK_OVERLAP}, separator='\\n\\n'(默认)",
) as step:
    # 2. 初始化固定大小分块器
    text_splitter = CharacterTextSplitter(
        chunk_size=CHUNK_SIZE,      # 每个块的大小（目标值，不是硬上限）
        chunk_overlap=CHUNK_OVERLAP # 块之间的重叠大小
    )

    # 3. 执行分块
    chunks = text_splitter.split_documents(docs)

    lens = [len(c.page_content) for c in chunks]
    over = [n for n in lens if n > CHUNK_SIZE]
    step.detail(f"块数：{len(chunks)}，长度 {min(lens)}~{max(lens)}，平均 {sum(lens)//len(lens)}")
    if over:
        step.detail(f"⚠ {len(over)} 块超过 chunk_size={CHUNK_SIZE}（见上方 LangChain 警告）")
        step.detail("  原因：它只在 '\\n\\n' 处切；单个段落本身就 >200 时无法再劈，只能原样保留")
    step.note(f"{len(chunks)} 块，{len(over)} 块超限")

# ---------------------------------------------------------------- 3. 打印
with stage.step("打印切分结果", note="原文输出，便于肉眼检查边界") as step:
    step.detail("以下为脚本原始输出（前 5 块）")

    # 4. 打印结果
    print(f"文本被切分为 {len(chunks)} 个块。\n")
    print("--- 前5个块内容示例 ---")
    for i, chunk in enumerate(chunks[:5]):
        print("=" * 60)
        # chunk 是一个 Document 对象，需要访问它的 .page_content 属性来获取文本
        print(f'块 {i+1} (长度: {len(chunk.page_content)}): "{chunk.page_content}"')

    step.note("完成")

stage.summary()

print()
print("=" * 72)
print("  观察点")
print("=" * 72)
print("1. 块 2 长度 201 > chunk_size 200 —— 固定大小分块并不保证大小上限")
print("2. 块边界落在 '\\n\\n'（空行）处，所以块长很参差（72 ~ 296）")
print("3. 短块（72 字符）会稀释检索精度：一个向量只代表很少的信息")
print()
print("→ 怎么改善？往下看 03_recursive_character_splitter.py")
print("=" * 72)
