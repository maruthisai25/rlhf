"""Build SFT data by letting the 9B teacher (llama-server) solve the training tasks.

Rejection sampling: each task is attempted up to --attempts times; the first trajectory that
passes the checker AND ends with a clean final message is kept. Output is TRL's conversational
tool-calling format: {"messages": [...], "tools": [...]}.

    python scripts/make_sft_data.py --out data/sft.jsonl --workers 4
"""
import argparse
import json
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from termbench.agent import ServerBackend, run_episode  # noqa: E402
from termbench.env import tool_schema  # noqa: E402
from termbench.tasks import load_tasks  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="tasks/train.jsonl")
ap.add_argument("--out", default="data/sft.jsonl")
ap.add_argument("--attempts", type=int, default=3)
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--temperature", type=float, default=0.5)
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--max-calls", type=int, default=6, help="drop trajectories with more tool calls than this")
a = ap.parse_args()

tasks = load_tasks(a.split)[: a.limit]
backend = ServerBackend(temperature=a.temperature, max_tokens=384, name="teacher")
tools = tool_schema()


def clean_messages(messages: list[dict]) -> list[dict]:
    out = []
    for m in messages:
        m = {k: v for k, v in m.items() if not k.startswith("_")}
        if m["role"] == "assistant" and m.get("tool_calls"):
            m["tool_calls"] = [{"type": "function", "function": tc["function"]} for tc in m["tool_calls"]]
            m["content"] = m.get("content") or ""
        out.append(m)
    return out


def solve(task):
    for attempt in range(a.attempts):
        ep = run_episode(task, backend)
        ok = ep.result["passed"] and ep.result["finished_cleanly"] and ep.result["n_calls"] <= a.max_calls and not ep.error
        print(f"{'KEEP' if ok else 'drop'} {task.id:32} attempt={attempt+1} calls={ep.result['n_calls']} passed={ep.result['passed']}"
              + (f" err={ep.error}" if ep.error else ""), flush=True)
        if ok:
            return task, ep
    return task, None


os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
kept, cats = 0, Counter()
with ThreadPoolExecutor(max_workers=a.workers) as ex, open(a.out, "w", encoding="utf-8") as f:
    for task, ep in ex.map(solve, tasks):
        if ep is None:
            continue
        f.write(json.dumps({"messages": clean_messages(ep.messages), "tools": tools, "task_id": task.id,
                            "category": task.category}, ensure_ascii=False) + "\n")
        kept += 1
        cats[task.category] += 1
print(f"kept {kept}/{len(tasks)} trajectories -> {a.out}")
print("per category:", dict(cats))
