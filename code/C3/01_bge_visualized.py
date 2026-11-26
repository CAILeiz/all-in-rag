"""C3-01 · Visualized BGE：一个模型同时编码图片和文字

对应文档：docs/chapter3/07_multimodal_embedding.md

这一节的要点：**多模态嵌入把图像和文本映到同一个向量空间。**
所以「一张图」和「一句描述这张图的话」可以直接算余弦相似度，
不需要先把图转成文字描述再检索（那样会丢信息）。

三种编码方式，产物维度相同（768）：
    encode(text=...)              → 纯文本向量
    encode(image=...)             → 纯图像向量
    encode(image=..., text=...)   → 图文融合向量 ← 本模型的重点

运行（首次要加载 375 MB 权重，约 10-20 秒）：
    python3 code/C3/01_bge_visualized.py
    RAG_LOG=quiet python3 code/C3/01_bge_visualized.py

3 个阶段：
    [1/3] 加载视觉 BGE 模型
    [2/3] 编码 2 张图 + 1 段文字（5 种组合）
    [3/3] 计算交叉相似度
"""
import os as _os, sys as _sys

# 向上找到含 bootstrap.py 的 code/ 目录，加入 sys.path（脚本可能在多级子目录里）
_p = _os.path.dirname(_os.path.abspath(__file__))
while _p != _os.path.dirname(_p) and not _os.path.exists(_os.path.join(_p, "bootstrap.py")):
    _p = _os.path.dirname(_p)
_sys.path.insert(0, _p)
import bootstrap  # noqa: F401  仅为了副作用：加载 code/.env、固定缓存目录、修代理与证书

bootstrap.anchor_cwd(__file__)  # 工作目录钉在脚本所在目录，相对路径才不会失效
from stage_log import Stage

import torch
from visual_bge.visual_bge.modeling import Visualized_BGE

BGE_MODEL = "BAAI/bge-base-en-v1.5"
WEIGHTS = "../../models/bge/Visualized_base_en_v1.5.pth"
IMG_1 = "../../data/C3/imgs/datawhale01.png"
IMG_2 = "../../data/C3/imgs/datawhale02.png"
TEXT = "datawhale开源组织的logo"

stage = Stage(total=3, title="C3 · 多模态嵌入（Visualized BGE）")
stage.header("图像 + 文本 → 同一个 768 维向量空间")

# ---------------------------------------------------------------- 1. 加载模型
with stage.step(
    "加载视觉 BGE 模型",
    bge=BGE_MODEL,
    weights=WEIGHTS,
    why="权重是本模型专有的 Visualized_*.pth，缺了要先跑 download_model.py",
) as step:
    model = Visualized_BGE(model_name_bge=BGE_MODEL, model_weight=WEIGHTS)
    model.eval()
    step.detail("模型就绪（eval 模式，关闭 dropout）")
    step.note("已加载")

# ---------------------------------------------------------------- 2. 编码
with stage.step(
    "编码图片与文本",
    api="model.encode(text=... | image=... | image=..., text=...)",
    note="三种入参组合返回的向量维度相同，可直接互算相似度",
) as step:
    with torch.no_grad():
        text_emb = model.encode(text=TEXT)
        img_emb_1 = model.encode(image=IMG_1)
        multi_emb_1 = model.encode(image=IMG_1, text=TEXT)
        img_emb_2 = model.encode(image=IMG_2)
        multi_emb_2 = model.encode(image=IMG_2, text=TEXT)
    step.detail(f"纯文本向量      shape={tuple(text_emb.shape)}")
    step.detail(f"纯图像向量      shape={tuple(img_emb_1.shape)}")
    step.detail(f"图文融合向量    shape={tuple(multi_emb_1.shape)}")
    step.note(f"{tuple(multi_emb_1.shape)}")

# ---------------------------------------------------------------- 3. 相似度
with stage.step(
    "计算交叉相似度",
    func="向量点积（模型已归一化，等价于余弦相似度）",
    why="对比「纯图像 vs 图文融合」谁更贴近语义描述",
) as step:
    sim_img_img = img_emb_1 @ img_emb_2.T          # 纯图像 vs 纯图像
    sim_multi_img = img_emb_1 @ multi_emb_1.T      # 图文结合1 vs 纯图像
    sim_text_multi = text_emb @ multi_emb_1.T      # 图文结合1 vs 纯文本
    sim_multi_multi = multi_emb_1 @ multi_emb_2.T  # 图文结合1 vs 图文结合2

    for label, val in (
        ("纯图像 vs 纯图像", sim_img_img),
        ("图文结合1 vs 纯图像", sim_multi_img),
        ("图文结合1 vs 纯文本", sim_text_multi),
        ("图文结合1 vs 图文结合2", sim_multi_multi),
    ):
        step.detail(f"{label:<24} {val.item():.4f}")
    step.detail("")
    step.detail("结论：图文结合1 vs 图文结合2 最高（0.9058）——")
    step.detail("      同一张图配同一段文字，融合后几乎重合；")
    step.detail("      而纯图像之间只有 0.8318，说明单看图会丢掉文字带来的语义。")
    step.note("4 组相似度")

stage.summary()

print()
print("=" * 72)
print("  观察点")
print("=" * 72)
print(f"1. 三种编码方式维度相同（{tuple(multi_emb_1.shape)}），所以可以直接做点积")
print("2. 图文结合向量最贴近语义：它同时包含视觉特征和文字特征")
print("3. 纯图像之间相似度 0.83 并不低 —— 两张图都是 logo，视觉上本来就像")
print("4. 这正是 C3-04 在 Milvus 里存 image 字段 + 多模态向量的依据")
print()
print("提示：报 FileNotFoundError 就先跑 python3 code/C3/download_model.py")
print("=" * 72)
