"""C2-01 · 用 Unstructured 解析 PDF，看清「文档元素」

对应文档：docs/chapter2/04_data_load.md

这一节的要点：**Unstructured 不是把 PDF 读成一坨文本，而是读成「带类型的元素」。**
每个元素有 category：Title / NarrativeText / ListItem / Table / Header / Footer ...
分块时就可以利用这些类型（比如「Table 单独成块」「Footer 丢弃」），
这比在纯文本上按长度硬切精准得多。

运行：
    python3 code/C2/01_unstructured_example.py
    RAG_LOG=quiet python3 code/C2/01_unstructured_example.py

依赖：unstructured + poppler（pdftoppm，用来把 PDF 转图片做 OCR）

3 个阶段：
    [1/3] 解析 PDF
    [2/3] 统计元素类型
    [3/3] 打印全部元素
"""

import os as _os, sys as _sys

# 向上找到含 bootstrap.py 的 code/ 目录，加入 sys.path（脚本可能在多级子目录里）
_p = _os.path.dirname(_os.path.abspath(__file__))
while _p != _os.path.dirname(_p) and not _os.path.exists(
    _os.path.join(_p, "bootstrap.py")
):
    _p = _os.path.dirname(_p)
_sys.path.insert(0, _p)
import bootstrap  # noqa: F401  仅为了副作用：加载 code/.env、固定缓存目录、修代理与证书

bootstrap.anchor_cwd(__file__)  # 工作目录钉在脚本所在目录，相对路径才不会失效
from stage_log import Stage

from unstructured.partition.auto import partition

# PDF文件路径
PDF_PATH = "../../data/C2/pdf/rag.pdf"

# 显式声明语言：不传时 unstructured 默认当英文（日志会打 "No languages specified"），
# 中文文档的句子切分/词性标注都会受影响。chi_sim=简体中文，eng=英文（URL、括号内容）。
LANGUAGES = ["chi_sim", "eng"]

stage = Stage(total=3, title="C2 · 数据加载")
stage.header("Unstructured.partition：PDF → 带类型的元素列表")

# ---------------------------------------------------------------- 1. 解析
with stage.step(
    "解析 PDF",
    api="partition(filename, content_type='application/pdf', languages=[...])",
    file=PDF_PATH,
    note="poppler 渲染页面、pdfminer 抽文字层；有文字层就不会调用 tesseract",
) as step:
    # 使用Unstructured加载并解析PDF文档
    # languages 不传 → 默认英文，中文 PDF 的切句会不准，所以显式指定
    elements = partition(
        filename=PDF_PATH,
        content_type="application/pdf",
        languages=LANGUAGES,
    )
    step.detail(
        f"解析完成：{len(elements)} 个元素，"
        f"共 {sum(len(str(e)) for e in elements)} 字符"
    )
    step.note(f"{len(elements)} 个元素")

# ---------------------------------------------------------------- 2. 统计类型
with stage.step("统计元素类型", why="不同类型在分块时有不同的处理策略") as step:
    # 统计元素类型
    from collections import Counter

    types = Counter(e.category for e in elements)

    for name, cnt in types.most_common():
        step.detail(f"{name:<20} {cnt:>4} 个")
    step.note(f"{len(types)} 种类型")

    step.detail("")
    step.detail("分块时的常见用法：")
    step.detail("  Title/NarrativeText → 按标题层级聚合，一个章节一块")
    step.detail("  Table               → 单独成块（别被切碎，否则表头丢失）")
    step.detail("  Header/Footer       → 直接丢弃（页码、版权声明是噪声）")

# ---------------------------------------------------------------- 3. 打印
with stage.step("打印全部元素", note="原文输出，元素很多时略长") as step:
    step.detail("以下为脚本原始输出")

    # 打印解析结果
    print(
        f"解析完成: {len(elements)} 个元素, {sum(len(str(e)) for e in elements)} 字符"
    )

    # 统计元素类型
    print(f"元素类型: {dict(types)}")

    # 显示所有元素
    print("\n所有元素:")
    for i, element in enumerate(elements, 1):
        print(f"Element {i} ({element.category}):")
        print(element)
        print("=" * 60)

    step.note("完成")

stage.summary()

print()
print("=" * 72)
print("  观察点")
print("=" * 72)
print(
    f"1. 279 个元素里 Title 占 {types.get('Title', 0)} 个 —— 大部分来自网页 PDF 的碎片标题"
)
print(f"2. NarrativeText 只有 {types.get('NarrativeText', 0)} 个 —— 真正的正文段落很少")
print("3. 这说明：PDF 解析出来的『元素』质量参差，不能直接当分块单位用")
print()
print("  → 实际项目里通常还要再清洗一遍（合并相邻短元素、丢 Footer、")
print("    把 Title 当章节标记），清洗逻辑就是 C2 第二节之后的内容。")
print()
print("提示：如果报 'Is poppler installed'，执行 brew install poppler")
print("=" * 72)

# ---- 诊断：查看一个元素的全部 key/value ----
import json


def dump_one(el, label=""):
    print(f"--- {label}{el.category} ---")
    print(json.dumps(el.to_dict(), ensure_ascii=False, indent=2))


dump_one(elements[0], "样例 · ")
dump_one(elements[17], "样例 · ")
dump_one(elements[86], "样例 · ")
dump_one(elements[96], "样例 · ")
print("\n字段覆盖情况：")
for ks, n in Counter(tuple(sorted(vars(e.metadata))) for e in elements).most_common():
    print(f"  {n:4d} 个 · {ks}")
