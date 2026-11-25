"""统一的 LLM 配置读取工具。

教程默认使用 AIHubmix（OpenAI 兼容）或 DeepSeek 官方接口；
本仓库跑通时更常用本机 Ollama 的 OpenAI 兼容端点。

用法（脚本里先做一次 sys.path 引导，再导入）：

    from llm_config import llm_kwargs, openai_llm_kwargs

    llm = ChatOpenAI(**llm_kwargs())
    Settings.llm = OpenAILike(**openai_llm_kwargs())

优先级：进程环境变量 > code/.env > 教程默认值（AIHubmix）。
配置项（可由 code/local_env.sh 一次性导出）：
    LLM_BASE_URL  默认 https://aihubmix.com/v1
    LLM_API_KEY   默认取 DEEPSEEK_API_KEY
    LLM_MODEL     默认 glm-4.7-flash-free
"""
import os
from pathlib import Path

DEFAULT_BASE_URL = "https://aihubmix.com/v1"
DEFAULT_MODEL = "glm-4.7-flash-free"

_CODE_DIR = Path(__file__).resolve().parent


def _load_dotenv_once() -> None:
    """读取 code/.env（不覆盖已存在的进程环境变量）。"""
    env_file = _CODE_DIR / ".env"
    if not env_file.exists():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(env_file, override=False)


def sanitize_proxy_env() -> None:
    """清掉会让 httpx/openai 直接报错的代理写法。

    - `ALL_PROXY=socks5://…`：httpx 没装 socksio，构造客户端时就抛
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


_load_dotenv_once()
sanitize_proxy_env()


def base_url() -> str:
    return os.getenv("LLM_BASE_URL") or DEFAULT_BASE_URL


def api_key() -> str:
    return os.getenv("LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY") or ""


def model_name() -> str:
    return os.getenv("LLM_MODEL") or DEFAULT_MODEL


# 兼容直接 `from llm_config import LLM_MODEL` 的写法（导入时取值）
LLM_BASE_URL = base_url()
LLM_API_KEY = api_key()
LLM_MODEL = model_name()


def llm_kwargs() -> dict:
    """langchain ChatOpenAI / ChatDeepSeek 通用参数。"""
    return {"model": model_name(), "api_key": api_key(), "base_url": base_url()}


def openai_llm_kwargs() -> dict:
    """llama_index OpenAILike / DeepSeek 参数。"""
    return {"model": model_name(), "api_key": api_key(), "api_base": base_url()}


def openai_client_kwargs() -> dict:
    """原生 openai.OpenAI 客户端参数。"""
    return {"api_key": api_key(), "base_url": base_url()}


def is_local_endpoint() -> bool:
    return any(h in base_url() for h in ("127.0.0.1", "localhost", "0.0.0.0"))
