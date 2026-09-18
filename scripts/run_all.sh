#!/usr/bin/env bash
# End-to-end pipeline, in order. Run inside WSL from the repo root, with the judge server up:
#   nohup setsid bash scripts/run_all.sh > logs/run_all.log 2>&1 &
# Each stage is skipped if its output already exists, so the script is re-runnable.
set -u
cd "$(dirname "$0")/.." || exit 1
PY=${PY:-~/rlhf/.venv/bin/python}
export JUDGE_URL=${JUDGE_URL:-http://172.20.128.1:8080}
export TERMBENCH_JUDGE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs results
log() { echo "[$(date +%H:%M:%S)] $*"; }
bench() {  # name split adapter-chain
  [ -f "results/$1.summary.json" ] && { log "skip benchmark $1 (exists)"; return; }
  log "benchmark $1  adapter=${3:-none}  split=$2"
  $PY -u scripts/run_benchmark.py --name "$1" --split "$2" ${3:+--adapter "$3"} --batch 32 > "logs/bench_$1.log" 2>&1
  grep -E '"pass_rate"|"mean_calls"|"mean_judge"' "logs/bench_$1.log"
}

# 0. tasks
[ -f tasks/test.jsonl ]       || $PY scripts/gen_tasks.py --validate
[ -f tasks/hard_test_raw.jsonl ] || $PY scripts/gen_tasks.py --hard --validate
[ -f tasks/grpo_mix.jsonl ]   || $PY -u scripts/filter_hard.py --workers 3 --attempts 2 > logs/filter_hard.log 2>&1

# 1. baseline
bench 0_base tasks/test.jsonl ""

# 2. SFT
[ -f data/sft.jsonl ] || $PY -u scripts/make_sft_data.py --workers 3 --attempts 3 > logs/make_sft_data.log 2>&1
[ -f runs/sft/final/adapter_config.json ] || $PY -u scripts/train_sft.py --out runs/sft > logs/train_sft.log 2>&1
bench 1_sft      tasks/test.jsonl      runs/sft/final
bench 1_sft_hard tasks/hard_test.jsonl runs/sft/final

# 3. GRPO (run 2 configuration: Dr. GRPO normalisation, pass-gated reward, small KL, hard+easy mix)
[ -f runs/grpo2/final/adapter_config.json ] || $PY -u scripts/train_grpo.py --adapter runs/sft/final \
    --split tasks/grpo_mix.jsonl --out runs/grpo2 --steps 40 --num-gens 8 --gen-batch 16 --micro-batch 2 \
    --judge-weight 0.3 --max-iters 6 --max-completion 768 --lr 1e-5 --beta 0.02 \
    --loss-type dr_grpo --scale-rewards none --reward-version v2 > logs/train_grpo2.log 2>&1
bench 3_grpo2      tasks/test.jsonl      runs/sft/final,runs/grpo2/final
bench 3_grpo2_hard tasks/hard_test.jsonl runs/sft/final,runs/grpo2/final

# 4. figures + table
$PY scripts/compare.py results
$PY scripts/make_figures.py
log "done"
