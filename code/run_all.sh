#!/usr/bin/env bash
# C1-C4 一键跑通脚本
#
#   bash code/run_all.sh              # 顺序跑全部
#   bash code/run_all.sh C2 C3        # 只跑指定章节
#
# 产物与日志：
#   code/logs/<章节>/<脚本名>.log
#   code/logs/summary.txt
set -u

CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$CODE_DIR/.." && pwd)"
# shellcheck disable=SC1091
source "$CODE_DIR/local_env.sh" >/dev/null

LOGS="$CODE_DIR/logs"
mkdir -p "$LOGS"
SUMMARY="$LOGS/summary.txt"
: > "$SUMMARY"

chapters=("$@")
if [ ${#chapters[@]} -eq 0 ]; then
  chapters=(C1 C2 C3 C4)
fi

declare -a ORDER=(
  "C1/01_langchain_example.py"
  "C1/02_llamaIndex_example.py"
  "C2/01_unstructured_example.py"
  "C2/02_character_splitter.py"
  "C2/03_recursive_character_splitter.py"
  "C2/04_semantic_chunker.py"
  "C3/01_bge_visualized.py"
  "C3/02_langchain_faiss.py"
  "C3/03_llamaindex_vector.py"
  "C3/04_multi_milvus.py"
  "C3/05_sentence_window_retrieval.py"
  "C3/06_recursive_retrieval.py"
  "C3/07_recursive_retrieval_v2.py"
  "C3/work_multimodal_dragon_search.py"
  "C3/work_hybrid_multimodal_search.py"
  "C4/01_hybrid_search.py"
  "C4/01_hybrid_search_v2.py"
  "C4/02_text_to_metadata_filter.py"
  "C4/03_text2sql_demo.py"
  "C4/03_text2sql_demo_v2.py"
  "C4/04_text_to_metadata_filter_v2.py"
  "C4/05_llm_based_routing.py"
  "C4/06_embedding_based_routing.py"
  "C4/07_rerank_and_refine.py"
  "C4/work_rerank_and_refine.py"
)

pass=0; fail=0; skip=0
for rel in "${ORDER[@]}"; do
  chap="${rel%%/*}"
  if ! printf '%s\n' "${chapters[@]}" | grep -qx "$chap"; then
    continue
  fi
  script="$CODE_DIR/$rel"
  logdir="$LOGS/$chap"
  mkdir -p "$logdir"
  log="$logdir/$(basename "${rel%.py}").log"
  echo "==> $rel"
  start=$(date +%s)
  ( cd "$(dirname "$script")" && python3 "$(basename "$script")" ) >"$log" 2>&1
  rc=$?
  dur=$(( $(date +%s) - start ))
  if [ $rc -eq 0 ]; then
    pass=$((pass+1)); status="PASS"
  else
    fail=$((fail+1)); status="FAIL"
  fi
  printf '%-8s %-45s %3ss  log=%s\n' "$status" "$rel" "$dur" "$log" | tee -a "$SUMMARY"
done

echo "" | tee -a "$SUMMARY"
echo "PASS=$pass FAIL=$fail" | tee -a "$SUMMARY"
[ $fail -eq 0 ]
