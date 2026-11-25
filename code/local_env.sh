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

echo "[local_env] REPO_ROOT=$REPO_ROOT"
echo "[local_env] LLM -> $LLM_BASE_URL (model=$LLM_MODEL)"
echo "[local_env] HF_HOME=$HF_HOME"
