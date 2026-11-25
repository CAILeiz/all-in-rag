#!/usr/bin/env bash
# curl 调 deepseek-chat 的最小脚本（OpenAI 兼容 /chat/completions）
#
#   bash code/curl_deepseek.sh "你的问题"
#   bash code/curl_deepseek.sh "你的问题" --raw      # 打印完整 JSON
#   bash code/curl_deepseek.sh --local "你好"        # 强制走本机 Ollama
#   bash code/curl_deepseek.sh --cloud "你好"        # 强制走 AIHubmix/DeepSeek
#
# 默认行为：读 code/.env 里的 LLM_BASE_URL / LLM_MODEL / LLM_API_KEY
#          —— 也就是当前这份配置里的本机 Ollama（model 别名 deepseek-chat）。
set -uo pipefail

CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- 读取配置（与 Python 脚本同一份 code/.env）----
if [ -f "$CODE_DIR/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  . "$CODE_DIR/.env"
  set +a
fi

BASE_URL="${LLM_BASE_URL:-https://api.deepseek.com}"
MODEL="${LLM_MODEL:-deepseek-chat}"
API_KEY="${LLM_API_KEY:-${DEEPSEEK_API_KEY:-}}"
QUESTION=""
RAW=0

while [ $# -gt 0 ]; do
  case "$1" in
    --local) BASE_URL="http://127.0.0.1:11434/v1"; MODEL="deepseek-chat"; API_KEY="ollama" ;;
    --cloud) BASE_URL="https://aihubmix.com/v1"; MODEL="glm-4.7-flash-free" ;;
    --raw)   RAW=1 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *)       QUESTION="$1" ;;
  esac
  shift
done

QUESTION="${QUESTION:-用一句话说明 RAG 是什么}"

# socks 代理会让 curl 之外的 httpx 报错，这里统一让本地请求绕开代理
case "$BASE_URL" in
  *127.0.0.1*|*localhost*) NO_PROXY="localhost,127.0.0.1" no_proxy="localhost,127.0.0.1" ;;
esac
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1}"

# JSON 转义：反斜杠、双引号
esc() { printf '%s' "$1" | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g'; }

PAYLOAD=$(cat <<JSON
{
  "model": "$MODEL",
  "messages": [
    {"role": "system", "content": "你是一个严谨的中文助手，回答简洁。"},
    {"role": "user", "content": "$(esc "$QUESTION")"}
  ],
  "temperature": 0.7,
  "max_tokens": 512,
  "stream": false
}
JSON
)

echo "POST $BASE_URL/chat/completions"
echo "model=$MODEL  question=$QUESTION"
echo "----"

RESP=$(curl -sS --max-time 120 \
  "$BASE_URL/chat/completions" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d "$PAYLOAD")

if [ $RAW -eq 1 ]; then
  printf '%s\n' "$RESP"
  exit 0
fi

# 优先用 python 解析，失败就原样打印（方便看 401/404 之类的报错体）
printf '%s' "$RESP" | python3 -c '
import json, sys
raw = sys.stdin.read()
try:
    d = json.loads(raw)
except Exception:
    print(raw)
    sys.exit(0)
if "error" in d:
    print("HTTP 错误:", json.dumps(d["error"], ensure_ascii=False))
    sys.exit(1)
print(d["choices"][0]["message"]["content"])
u = d.get("usage") or {}
p, c, t = u.get("prompt_tokens"), u.get("completion_tokens"), u.get("total_tokens")
print("")
print("[tokens] prompt=%s completion=%s total=%s" % (p, c, t))
'
