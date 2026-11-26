"""C3 · 下载 Visualized BGE 权重（C3-01/04 及 work 脚本的前置条件）

对应文档：docs/chapter3/07_multimodal_embedding.md 4.1 节「步骤2：下载模型权重」

权重文件不进 Git（375 MB），所以第一次跑 C3 的多模态脚本前要先执行这个。

    python3 code/C3/download_model.py

已经下载过会直接跳过，只打印文件大小。
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

import sys
from pathlib import Path

import requests

MODEL_DIR = Path("../../models/bge")
MODEL_FILE = MODEL_DIR / "Visualized_base_en_v1.5.pth"
DOWNLOAD_URL = (
    "https://huggingface.co/BAAI/bge-visualized/resolve/main/"
    "Visualized_base_en_v1.5.pth?download=true"
)


def _fetch(model_file: Path, step) -> None:
    """流式下载，每 10% 打一行进度（避免把日志刷掉）。"""
    response = requests.get(DOWNLOAD_URL, stream=True, timeout=60)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    downloaded = 0
    last_pct = -10

    with open(model_file, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if not chunk:
                continue
            f.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                pct = int(downloaded / total_size * 100)
                if pct >= last_pct + 10:
                    last_pct = pct
                    step.detail(
                        f"下载中 {pct:>3}%  "
                        f"({downloaded / 1048576:.0f}/{total_size / 1048576:.0f} MB)"
                    )


def download_visualized_bge_model() -> str | None:
    """下载 Visual BGE 权重；已存在则直接复用。返回文件路径或 None。"""
    stage = Stage(total=1, title="C3 · 下载 Visualized BGE 权重")
    stage.header("375 MB，只需下载一次；已存在会自动跳过")

    result = None
    with stage.step(
        "准备模型权重",
        path=str(MODEL_FILE),
        url="huggingface.co/BAAI/bge-visualized",
        why="C3 的多模态脚本都依赖这个权重，缺了会报 FileNotFoundError",
    ) as step:
        if MODEL_FILE.exists():
            size_mb = MODEL_FILE.stat().st_size / 1048576
            step.detail(f"模型文件已存在，跳过下载（{size_mb:.1f} MB）")
            step.note(f"已存在 {size_mb:.0f} MB")
            result = str(MODEL_FILE)
        else:
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            step.detail(f"目标目录：{MODEL_DIR}")
            try:
                _fetch(MODEL_FILE, step)
                size_mb = MODEL_FILE.stat().st_size / 1048576
                step.detail(f"下载完成，文件大小 {size_mb:.1f} MB")
                step.note(f"已下载 {size_mb:.0f} MB")
                result = str(MODEL_FILE)
            except requests.exceptions.RequestException as e:
                step.detail(f"下载失败：{e}")
                # 删掉不完整文件，否则下次会被误判成「已存在」
                MODEL_FILE.unlink(missing_ok=True)
            except Exception as e:
                step.detail(f"发生错误：{e}")
                MODEL_FILE.unlink(missing_ok=True)

    stage.summary()
    return result


if __name__ == "__main__":
    model_path = download_visualized_bge_model()
    print()
    if model_path:
        print(f"✅ 模型准备就绪：{model_path}")
        print("   接下来可以跑：python3 code/C3/01_bge_visualized.py")
    else:
        print("❌ 模型下载失败")
        print("   检查网络/代理，或手动下载后放到 models/bge/：")
        print(f"   {DOWNLOAD_URL}")
    sys.exit(0 if model_path else 1)
