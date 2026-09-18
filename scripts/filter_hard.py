"""Keep only composite tasks the 9B teacher can solve (>= 1 of N attempts), so the RL set is
hard-but-solvable. Also writes the GRPO training mix (hard + a slice of easy tasks).

    python scripts/filter_hard.py --workers 3 --attempts 2
"""
import argparse
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from termbench.agent import ServerBackend, run_episode  # noqa: E402
from termbench.tasks import load_tasks, save_tasks  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--attempts", type=int, default=2)
ap.add_argument("--workers", type=int, default=3)
ap.add_argument("--easy", type=int, default=64, help="easy train tasks mixed into the GRPO set")
ap.add_argument("--max-hard-train", type=int, default=112)
a = ap.parse_args()

backend = ServerBackend(temperature=0.5, max_tokens=384, name="teacher")


def solvable(task):
    for _ in range(a.attempts):
        ep = run_episode(task, backend)
        if ep.result["passed"]:
            print(f"KEEP {task.id:40} calls={ep.result['n_calls']}", flush=True)
            return task, True
    print(f"drop {task.id:40}", flush=True)
    return task, False


for split in ("hard_train", "hard_test"):
    raw = load_tasks(f"tasks/{split}_raw.jsonl")
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        kept = [t for t, ok in ex.map(solvable, raw) if ok]
    if split == "hard_train":
        kept = kept[: a.max_hard_train]
    save_tasks(kept, f"tasks/{split}.jsonl")
    print(f"{split}: kept {len(kept)}/{len(raw)}")

easy = load_tasks("tasks/train.jsonl")
random.Random(0).shuffle(easy)
mix = load_tasks("tasks/hard_train.jsonl") + easy[: a.easy]
random.Random(1).shuffle(mix)
save_tasks(mix, "tasks/grpo_mix.jsonl")
print(f"grpo_mix: {len(mix)} tasks")
