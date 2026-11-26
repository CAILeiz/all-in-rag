"""脚本级环境引导：加载 code/.env、修正代理与证书变量。

必须在任何重依赖（transformers / sentence_transformers / huggingface_hub /
langchain / llama_index / openai）之前导入，否则 HF_HOME、NLTK_DATA、SSL_CERT_FILE
等变量会太晚生效。

用法（放在脚本最顶部）：

    import os as _os, sys as _sys
    _p = _os.path.dirname(_os.path.abspath(__file__))
    while _p != _os.path.dirname(_p) and not _os.path.exists(_os.path.join(_p, "bootstrap.py")):
        _p = _os.path.dirname(_p)
    _sys.path.insert(0, _p)
    import bootstrap  # noqa: F401  # 仅为了副作用：加载 .env、修代理
    bootstrap.anchor_cwd(__file__)   # 把工作目录钉在脚本所在目录

`anchor_cwd` 解决的是「脚本里写了 ../../data/xxx 这种相对路径」的问题：
不管你在仓库根目录、code/ 还是 code/C1 下执行，相对路径都能正确解析。
"""
import os
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
ENV_FILE = CODE_DIR / ".env"


def load_env(path: Path | None = None) -> None:
    """加载 code/.env（不覆盖已存在的进程环境变量）。"""
    env_file = path or ENV_FILE
    if not env_file.exists():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(env_file, override=False)


def sanitize_proxy_env() -> None:
    """清掉会让 httpx/openai 直接报错的代理写法。

    - `ALL_PROXY=socks5://…`：httpx 未装 socksio 时构造客户端就抛
      ImportError: Using SOCKS proxy, but the 'socksio' package is not installed。
      只删 socks 版 ALL_PROXY，http_proxy/https_proxy 保留，云端接口仍可走代理。
    - `NO_PROXY` 里写成 `[::1]`：httpx 解析 no_proxy 时会抛
      InvalidURL: Invalid port: ':1]'，这里把带方括号的项剔除。
    """
    for var in ("ALL_PROXY", "all_proxy"):
        if os.environ.get(var, "").lower().startswith("socks"):
            os.environ.pop(var, None)

    for var in ("NO_PROXY", "no_proxy"):
        val = os.environ.get(var)
        if val and "[" in val:
            kept = [h for h in val.split(",") if h and "[" not in h]
            os.environ[var] = ",".join(kept)


def ensure_ssl_cert() -> None:
    """把 SSL 根证书指向 certifi（本机 Python 3.10 找不到系统 CA，requests 会报错）。"""
    if os.environ.get("SSL_CERT_FILE"):
        return
    try:
        import certifi
    except ImportError:
        return
    cafile = certifi.where()
    if os.path.exists(cafile):
        os.environ.setdefault("SSL_CERT_FILE", cafile)
        os.environ.setdefault("REQUESTS_CA_BUNDLE", cafile)
        os.environ.setdefault("CURL_CA_BUNDLE", cafile)


def ensure_data_dirs() -> None:
    """把 HF / NLTK 等缓存固定到仓库内 .cache/，并确保库真的会去那里找。

    这一步是「脚本能独立跑」的关键：
    - env 变量只影响**之后**才 import 的库，所以要在重依赖之前执行；
    - 光设 NLTK_DATA 环境变量还不够 —— 已经 import 过 nltk 时它的 data.path 是快照，
      必须再往 nltk.data.path 里显式插一条，否则找不到已下载的 punkt_tab。
    """
    cache_dir = CODE_DIR.parent / ".cache"
    for sub, var in (
        ("hf", "HF_HOME"),
        ("nltk", "NLTK_DATA"),
        ("llama_index", "LLAMA_INDEX_CACHE_DIR"),
        ("chroma", "CHROMA_CACHE_DIR"),
        ("matplotlib", "MPLCONFIGDIR"),
    ):
        path = cache_dir / sub
        # 已由 local_env.sh / .env 指定的路径优先，这里只做兜底
        os.environ.setdefault(var, str(path))

    nltk_dir = os.environ["NLTK_DATA"]
    try:
        import nltk.data
    except ImportError:
        return  # 没装 nltk，等真正用到时再说
    if nltk_dir not in nltk.data.path:
        nltk.data.path.insert(0, nltk_dir)


# unstructured 解析 PDF 时必需、且会静默吞掉异常的 NLTK 资源。
# 缺任意一个的后果不是清晰的 LookupError，而是「文字层判定失败 → 回退 OCR →
# 报 TesseractNotFoundError」，极具误导性，所以这里主动补齐。
_NLTK_REQUIRED = (
    ("punkt_tab", "tokenizers/punkt_tab"),
    ("averaged_perceptron_tagger_eng", "taggers/averaged_perceptron_tagger_eng"),
    ("punkt", "tokenizers/punkt"),
    ("stopwords", "corpora/stopwords"),
    ("omw-1.4", "corpora/omw-1.4.zip"),  # 这个包下载后是 zip，不解压
)


def ensure_nltk_data(verbose: bool = False) -> None:
    """检查 NLTK 资源，缺的自动下载（含代理与 SSL 修正）。

    只在真的缺失时才联网；全部就绪时只做几次本地查找，几乎零开销。
    没有网络时静默跳过，让原始报错暴露出来。
    """
    try:
        import nltk
        import nltk.data
    except ImportError:
        return

    def present(probe: str) -> bool:
        try:
            nltk.data.find(probe)
            return True
        except LookupError:
            return False

    missing = [pkg for pkg, probe in _NLTK_REQUIRED if not present(probe)]
    if not missing:
        return

    os.environ.setdefault("NLTK_ALLOW_PROXIED_URLOPEN", "1")
    target = os.environ["NLTK_DATA"]
    if verbose:
        print(f"[bootstrap] NLTK 缺少 {len(missing)} 个资源，开始下载：{', '.join(missing)}")
    for pkg in missing:
        try:
            ok = nltk.download(pkg, download_dir=target, quiet=True)
            if verbose:
                print(f"[bootstrap]   {pkg:32} {'ok' if ok else 'FAILED'}")
        except Exception:
            pass  # 下载失败就保持原样，让真实错误在调用方暴露


load_env()
sanitize_proxy_env()
ensure_ssl_cert()
ensure_data_dirs()
ensure_nltk_data(verbose=bool(os.environ.get("RAG_VERBOSE")))




def anchor_cwd(script_file: str, *, quiet: bool = True) -> Path:
    """把当前工作目录切到脚本所在目录。

    教程脚本里大量使用 `../../data/C1/xxx` 这类相对路径，只有在 code/Cx 下执行才对。
    这里统一锚定，避免「换个目录就 File does not exist」。

    同时也会把仓库根目录挂到 REPO_ROOT 环境变量，方便脚本按绝对路径拼。

    Returns:
        脚本所在目录的绝对路径。
    """
    script_dir = Path(script_file).resolve().parent
    old = Path.cwd()
    if old != script_dir:
        os.chdir(script_dir)
        if not quiet:
            print(f"[bootstrap] 工作目录已切到 {script_dir}")
    os.environ.setdefault("REPO_ROOT", str(CODE_DIR.parent))
    return script_dir


if __name__ == "__main__":
    print("CODE_DIR =", CODE_DIR)
    for key in ("LLM_BASE_URL", "LLM_MODEL", "HF_HOME", "NLTK_DATA", "SSL_CERT_FILE", "NO_PROXY"):
        print(f"{key} = {os.environ.get(key)}")
    sys.exit(0)
