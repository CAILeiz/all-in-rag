#!/usr/bin/env bash
# 本地一键环境（不依赖任何云端 API Key）
#
#   source code/local_env.sh          # 之后可在任意目录运行脚本
#   bash code/run_all.sh              # 顺序跑 C1~C4
#
# 它做四件事：
#   1) 读取 code/.env（LLM 端点、缓存目录等配置的唯一来源）
#   2) 把大模型指向本机 Ollama 的 OpenAI 兼容端点
#   3) 把 HF / NLTK / llama_index 缓存固定到仓库内 .cache/（不污染 ~/Library/Caches）
#   4) 修掉本机代理与 SSL 证书的两个坑（见文件末尾注释）
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
export REPO_ROOT

ENV_FILE="$REPO_ROOT/code/.env"
if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  . "$ENV_FILE"
  set +a
fi

# ---- 大模型：本机 Ollama ----
export OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://127.0.0.1:11434/v1}"
export LLM_BASE_URL="${LLM_BASE_URL:-$OLLAMA_BASE_URL}"
export LLM_MODEL="${LLM_MODEL:-deepseek-chat}"     # ollama 里的别名，指向 qwen2.5 系列
export LLM_API_KEY="${LLM_API_KEY:-ollama}"

# 教程脚本里写死读 DEEPSEEK_* 的地方也一并指向本地
export DEEPSEEK_API_BASE="$LLM_BASE_URL"
export DEEPSEEK_API_KEY="$LLM_API_KEY"

# ---- 缓存目录（按仓库实际位置动态计算，优先于 .env 里的绝对路径）----
export HF_HOME="$REPO_ROOT/.cache/hf"
export NLTK_DATA="$REPO_ROOT/.cache/nltk"
export LLAMA_INDEX_CACHE_DIR="$REPO_ROOT/.cache/llama_index"
export CHROMA_CACHE_DIR="$REPO_ROOT/.cache/chroma"
export AUTO_DOWNLOAD_NLTK="False"

# ---- 坑 1：socks5 版 ALL_PROXY ----
# httpx 没装 socksio，构造 openai 客户端时会直接抛
# ImportError: Using SOCKS proxy, but the 'socksio' package is not installed
if [ -n "${ALL_PROXY:-}" ]; then
  # macOS 自带 bash 3.2 不支持 ${var,,}，用 tr 转小写
  case "$(printf '%s' "$ALL_PROXY" | tr '[:upper:]' '[:lower:]')" in
    socks*) export ALL_PROXY="" all_proxy="" ;;
  esac
fi

# ---- 坑 2：NO_PROXY 里的 [::1] 写法 ----
# httpx 解析 no_proxy 会抛 InvalidURL: Invalid port: ':1]'
export NO_PROXY="localhost,127.0.0.1,0.0.0.0"
export no_proxy="$NO_PROXY"

# ---- 坑 3：SSL 根证书 ----
# 本机 Python 3.10 找不到系统 CA，bilibili / requests 会报 CERTIFICATE_VERIFY_FAILED
_CERTIFI="$(python3 -c 'import certifi;print(certifi.where())' 2>/dev/null)"
if [ -n "$_CERTIFI" ] && [ -f "$_CERTIFI" ]; then
  export SSL_CERT_FILE="$_CERTIFI"
  export REQUESTS_CA_BUNDLE="$_CERTIFI"
  export CURL_CA_BUNDLE="$_CERTIFI"
fi

# ---- NLTK 语料：unstructured 解析 PDF 的硬依赖 ----
# 缺资源时 unstructured 会**静默**吞掉 LookupError，让文字层判定失败，
# 然后回退到 OCR 分支报 "tesseract is not installed"（极具误导性的报错）。
# 真正缺的是下面这几个，这里做一次「缺失才补」，已存在则瞬间返回。
ensure_nltk() {
  # 新版 nltk 默认禁止走代理下载；本机常带代理，放开这一道保护
  export NLTK_ALLOW_PROXIED_URLOPEN=1
  _NLTK_PY="$(command -v python3)" || return 0
  "$_NLTK_PY" - "$NLTK_DATA" <<'PY' 2>/dev/null || true
import os, sys
data_dir = sys.argv[1]
os.environ["NLTK_DATA"] = data_dir
try:
    import nltk
    import nltk.data
except ImportError:
    sys.exit(0)   # 没装 nltk 就别管，等真正用到时再说
nltk.data.path.insert(0, data_dir)

NEEDED = ["punkt_tab", "averaged_perceptron_tagger_eng", "punkt", "stopwords", "omw-1.4"]

# 落盘路径 → 用 nltk.data.find 判断是否真的存在（只读，不触发下载）
# 注意：有的包下载后是 zip（如 omw-1.4），探针要写成 .zip 才查得到
PROBE = {
    "punkt_tab": "tokenizers/punkt_tab",
    "averaged_perceptron_tagger_eng": "taggers/averaged_perceptron_tagger_eng",
    "punkt": "tokenizers/punkt",
    "stopwords": "corpora/stopwords",
    "omw-1.4": "corpora/omw-1.4.zip",
}

def present(pkg: str) -> bool:
    """已存在返回 True；缺了返回 False。纯查找，绝不下载。"""
    probe = PROBE.get(pkg)
    if not probe:
        return False
    try:
        nltk.data.find(probe)
        return True
    except LookupError:
        return False

missing = [p for p in NEEDED if not present(p)]
if missing:
    print(f"[local_env] NLTK 缺失 {len(missing)} 个资源，正在补下：{', '.join(missing)}")
    for pkg in missing:
        ok = nltk.download(pkg, download_dir=data_dir, quiet=True)
        print(f"[local_env]   {pkg:32} {'ok' if ok else 'FAILED'}")
PY
}

mkdir -p "$NLTK_DATA" "$HF_HOME"
# 有网才下载；离线时跳过，不拖慢 source
ensure_nltk

echo "[local_env] REPO_ROOT=$REPO_ROOT"
echo "[local_env] LLM -> $LLM_BASE_URL (model=$LLM_MODEL)"
echo "[local_env] HF_HOME=$HF_HOME"
echo "[local_env] NLTK_DATA=$NLTK_DATA"
