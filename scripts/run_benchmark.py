"""Benchmark a model on a task split.

    # base model
    python scripts/run_benchmark.py --name base --model Qwen/Qwen3.5-0.8B
    # with a LoRA adapter
    python scripts/run_benchmark.py --name sft --model Qwen/Qwen3.5-0.8B --adapter runs/sft/final
    # the 9B teacher through llama-server (ceiling / sanity)
    python scripts/run_benchmark.py --name teacher --backend server --workers 4
    # also ask the judge to score every transcript (reported, not added to reward)
    TERMBENCH_JUDGE=1 python scripts/run_benchmark.py ...
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from termbench.benchmark import run_benchmark  # noqa: E402
from termbench.tasks import load_tasks  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--split", default="tasks/test.jsonl")
ap.add_argument("--backend", choices=["hf", "server"], default="hf")
ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
ap.add_argument("--adapter", default=None, help="LoRA adapter, or a comma-separated chain merged in order "
                "(GRPO adapters need the SFT adapter first: runs/sft/final,runs/grpo2/final)")
ap.add_argument("--temperature", type=float, default=0.7)
ap.add_argument("--max-new-tokens", type=int, default=384)
ap.add_argument("--max-turns", type=int, default=None)
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--workers", type=int, default=1, help="threads for the server backend")
ap.add_argument("--batch", type=int, default=8, help="episodes generated per batch for the hf backend")
ap.add_argument("--out", default="results")
ap.add_argument("--verbose", action="store_true")
a = ap.parse_args()

tasks = load_tasks(a.split)
if a.limit:
    tasks = tasks[: a.limit]

if a.backend == "hf":
    from termbench.agent import HFBackend

    backend = HFBackend(a.model, adapter=a.adapter, temperature=a.temperature, max_new_tokens=a.max_new_tokens)
    if a.workers != 1:
        print("hf backend is sequential; forcing --workers 1")
        a.workers = 1
else:
    from termbench.agent import ServerBackend

    backend = ServerBackend(temperature=a.temperature, max_tokens=a.max_new_tokens, name="llama-server:" + os.environ.get("JUDGE_MODEL", "judge"))

run_benchmark(tasks, backend, a.out, a.name, workers=a.workers, verbose=a.verbose, max_turns=a.max_turns,
              batch=a.batch if a.backend == "hf" and not a.verbose else 1)
