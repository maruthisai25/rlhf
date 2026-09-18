"""Run the agent over a task split and write per-episode records plus a summary."""
from __future__ import annotations

import json
import statistics
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .agent import Episode, run_episode, run_episodes_batched
from .tasks import Task


def summarize(episodes: list[Episode]) -> dict:
    by_cat: dict[str, list[Episode]] = defaultdict(list)
    for e in episodes:
        by_cat[e.category].append(e)
    passed = [e.result["passed"] for e in episodes]
    judged = [e.result["judge"] for e in episodes if e.result.get("judge") is not None]
    summary = {
        "n": len(episodes),
        "pass_rate": round(sum(passed) / max(1, len(passed)), 4),
        "mean_reward": round(statistics.fmean(e.result["reward"] for e in episodes), 4) if episodes else 0,
        "mean_calls": round(statistics.fmean(e.result["n_calls"] for e in episodes), 2) if episodes else 0,
        "finished_cleanly_rate": round(statistics.fmean(e.result["finished_cleanly"] for e in episodes), 4) if episodes else 0,
        "errors": sum(1 for e in episodes if e.error),
        "mean_judge": round(statistics.fmean(judged), 4) if judged else None,
        "seconds": round(sum(e.seconds for e in episodes), 1),
        "by_category": {
            c: {"n": len(v), "pass_rate": round(sum(e.result["passed"] for e in v) / len(v), 3)}
            for c, v in sorted(by_cat.items())
        },
        "by_difficulty": {},
    }
    return summary


def run_benchmark(tasks: list[Task], backend, out_dir: str | Path, name: str, workers: int = 1,
                  verbose: bool = False, max_turns: int | None = None, batch: int = 1) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    episodes: list[Episode] = []

    def report(ep: Episode):
        status = "PASS" if ep.result["passed"] else "fail"
        print(f"{status:4} {ep.task_id:32} calls={ep.result['n_calls']:2} r={ep.result['reward']:+.2f} {ep.seconds:5.1f}s"
              + (f"  ERR {ep.error}" if ep.error else ""), flush=True)

    def one(task: Task) -> Episode:
        ep = run_episode(task, backend, max_turns=max_turns, verbose=verbose)
        report(ep)
        return ep

    if batch > 1 and hasattr(backend, "step_batch"):
        episodes = run_episodes_batched(tasks, backend, batch_size=batch, max_turns=max_turns, on_done=report)
    elif workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            episodes = list(ex.map(one, tasks))
    else:
        episodes = [one(t) for t in tasks]

    with open(out_dir / f"{name}.episodes.jsonl", "w", encoding="utf-8") as f:
        for e in episodes:
            f.write(json.dumps(e.to_dict(), ensure_ascii=False) + "\n")
    summary = summarize(episodes)
    summary["name"] = name
    summary["backend"] = getattr(backend, "name", str(type(backend).__name__))
    summary["wall_seconds"] = round(time.time() - t0, 1)
    with open(out_dir / f"{name}.summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "by_category"}, indent=2))
    print("by category:", json.dumps(summary["by_category"]))
    return summary
