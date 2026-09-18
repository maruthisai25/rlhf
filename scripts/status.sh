#!/usr/bin/env bash
# Quick progress report for background jobs. Run inside WSL from the repo root.
cd "$(dirname "$0")/.." || exit 1
for f in logs/bench_*.log; do
  [ -f "$f" ] || continue
  done=$(grep -cE '^(PASS|fail)' "$f"); pass=$(grep -c '^PASS' "$f")
  echo "$f: $done done, $pass pass"
done
if [ -f logs/make_sft_data.log ]; then
  echo "make_sft_data: $(grep -c KEEP logs/make_sft_data.log) kept, $(grep -c drop logs/make_sft_data.log) dropped"
fi
for f in logs/train_*.log; do
  [ -f "$f" ] || continue
  echo "== $f (tail) =="; tail -n 5 "$f"
done
echo "gpu: $(/usr/lib/wsl/lib/nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader)"
