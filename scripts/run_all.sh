#!/usr/bin/env bash
# End-to-end pipeline, in order. Run inside WSL from the repo root, with the judge server up:
#   nohup setsid bash scripts/run_all.sh > logs/run_all.log 2>&1 &
#
# Every stage is skipped if its output already exists. In a fresh clone all outputs are
# committed, so nothing runs; to actually reproduce, wipe the derived artefacts first:
#   FRESH=1 nohup setsid bash scripts/run_all.sh > logs/run_all.log 2>&1 &
# Note that FRESH re-samples the teacher-filtered composite splits, so "hard" numbers will
# differ slightly from the committed ones; the single-step splits are seeded and identical.
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

if [ "${FRESH:-0}" = "1" ]; then
  log "FRESH=1: removing derived artefacts"
  rm -rf tasks/*.jsonl data/sft.jsonl runs/*/final runs/*/completions results/*.json results/*.jsonl
  find logs -name '*.log' ! -name run_all.log -delete
fi

# 0. tasks
[ -f tasks/test.jsonl ]          || $PY scripts/gen_tasks.py --validate
[ -f tasks/hard_test_raw.jsonl ] || $PY scripts/gen_tasks.py --hard --validate
[ -f tasks/grpo_mix.jsonl ]      || $PY -u scripts/filter_hard.py --workers 3 --attempts 2 > logs/filter_hard.log 2>&1

# 1. baseline
bench 0_base      tasks/test.jsonl      ""
bench 0_base_hard tasks/hard_test.jsonl ""

# 2. SFT
[ -f data/sft.jsonl ] || $PY -u scripts/make_sft_data.py --workers 3 --attempts 3 > logs/make_sft_data.log 2>&1
[ -f runs/sft/final/adapter_config.json ] || $PY -u scripts/train_sft.py --out runs/sft > logs/train_sft.log 2>&1
bench 1_sft      tasks/test.jsonl      runs/sft/final
bench 1_sft_hard tasks/hard_test.jsonl runs/sft/final

# 3a. GRPO run 1 (the failure case in the report): single-step prompts, group-std scaling,
#     reward v1, no KL, lr 2e-5. Kept in the recipe as the control arm for run 2.
[ -f runs/grpo/final/adapter_config.json ] || $PY -u scripts/train_grpo.py --adapter runs/sft/final \
    --split tasks/train.jsonl --out runs/grpo --steps 50 --num-gens 8 --gen-batch 16 --micro-batch 2 \
    --judge-weight 0.3 --max-iters 6 --max-completion 1024 --lr 2e-5 --beta 0 \
    --loss-type dapo --scale-rewards group --reward-version v1 > logs/train_grpo.log 2>&1
bench 2_grpo      tasks/test.jsonl      runs/sft/final,runs/grpo/final
bench 2_grpo_hard tasks/hard_test.jsonl runs/sft/final,runs/grpo/final

# 3b. GRPO run 2: hard+easy mix, Dr. GRPO normalisation, pass-gated reward v2, KL 0.02, lr 1e-5
[ -f runs/grpo2/final/adapter_config.json ] || $PY -u scripts/train_grpo.py --adapter runs/sft/final \
    --split tasks/grpo_mix.jsonl --out runs/grpo2 --steps 40 --num-gens 8 --gen-batch 16 --micro-batch 2 \
    --judge-weight 0.3 --max-iters 6 --max-completion 768 --lr 1e-5 --beta 0.02 \
    --loss-type dr_grpo --scale-rewards none --reward-version v2 > logs/train_grpo2.log 2>&1
bench 3_grpo2      tasks/test.jsonl      runs/sft/final,runs/grpo2/final
bench 3_grpo2_hard tasks/hard_test.jsonl runs/sft/final,runs/grpo2/final

# 4. table + figures (make_figures refuses to run with any stage missing)
$PY scripts/compare.py results
$PY scripts/make_figures.py
log "done"
