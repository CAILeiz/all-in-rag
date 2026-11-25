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


load_env()
sanitize_proxy_env()
ensure_ssl_cert()




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
